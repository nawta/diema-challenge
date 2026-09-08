""" exp067: post-hoc calibration of the 7-way equal-weight ensemble.

Compares three calibration methods (and the no-calibration baseline) on the
fixed 7-way OOF logits, using nested LPO so calibration parameters are never
fit and evaluated on the same fold.

Methods
-------
- ``baseline``       : no calibration (equal-weight 7-way logit mean as-is)
- ``temperature``    : single scalar temperature ``T`` (1 param), NLL fit
- ``uniform_prior``  : analytic Bayes adjust assuming uniform target prior
                      (0 fitted params; bias = log of model marginal prediction)
- ``class_bias_l2``  : 12-D additive class-bias with L2, lambda chosen by
                      inner 9-fold LPO

Statistical protocol (matches §5.1 / §11.1):
- Outer 10-fold LPO (existing fold structure on disk).
- Inner 9-fold LPO (each inner fold = one outer-train performer group) for
  hyperparameter selection (only ``class_bias_l2``).
- ``--num-seeds`` controls the optimisation init seed (does NOT retrain models).
- Sample-level paired bootstrap on the 7992 OOF samples (1000 resamples by
  default) for the ``calibrated F1 - baseline F1`` distribution.
- For ``class_bias_l2``: direction-consistency check across nested-LPO fits
  vs. the full-data fit (overfit guard).

See: §5.1,
Related: tools/ensemble_models.py (raw OOF averaging),
         tools/infer_test_ensemble.py (test inference with the same 7 members).
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.env import EnvConfig  # noqa: E402
from diema.data.parser import IDX_TO_EMOTION, parse_performer_id  # noqa: E402

# Production 7-way ensemble. Order matches tools/infer_test_ensemble.py /
# output/predictions/ensemble_7way/.
DEFAULT_ENSEMBLE = [
    "exp002_a04_smooth",
    "exp003_ctrgcn_a00",
    "exp004_skateformer_a01",
    "exp006_protogcn_a00",
    "exp020_conv1d_transformer_a01",
    "exp023_keypoint_pool_mlp_a00",
    "exp034_regionaware_convtr_a00",
]
NUM_CLASSES = 12
NUM_FOLDS = 10

# Confusion-driven priority classes (plan §5.1).
PRIORITY_CLASSES = ("guilt", "sadness", "pride")


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def log_sum_exp(z: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """Numerically stable log-sum-exp."""
    m = z.max(axis=axis, keepdims=True)
    out = m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))
    return out if keepdims else out.squeeze(axis)


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    m = z.max(axis=axis, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=axis, keepdims=True)


def nll_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean cross-entropy of ``logits`` against integer ``labels``."""
    log_probs = logits - log_sum_exp(logits, axis=-1, keepdims=True)
    return float(-log_probs[np.arange(len(labels)), labels].mean())


# ---------------------------------------------------------------------------
# Calibration methods
# ---------------------------------------------------------------------------

def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    init: float = 1.0,
    bounds: tuple[float, float] = (1e-3, 100.0),
) -> float:
    """Fit a single positive temperature ``T`` by minimising NLL.

    ``logits / T`` is the calibrated logit; T<1 sharpens, T>1 softens.
    """

    def obj(x: np.ndarray) -> tuple[float, np.ndarray]:
        T = float(x[0])
        scaled = logits / T
        log_probs = scaled - log_sum_exp(scaled, axis=-1, keepdims=True)
        nll = -log_probs[np.arange(len(labels)), labels].mean()
        # d/dT NLL = (1/T^2) * mean(z[y] - E_p[z])
        # (derive: d/dT log p[y] = (E_p[z] - z[y])/T^2 → flip sign for NLL)
        p = softmax(scaled, axis=-1)
        ez = (p * logits).sum(axis=1)  # E_p[z] per sample
        zy = logits[np.arange(len(labels)), labels]
        grad = float(((zy - ez).mean()) / (T * T))
        return float(nll), np.array([grad], dtype=np.float64)

    res = minimize(
        obj,
        x0=np.array([init], dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=[bounds],
        options={"gtol": 1e-9, "ftol": 1e-12, "maxiter": 200},
    )
    if not res.success:
        warnings.warn(f"fit_temperature L-BFGS-B did not converge: {res.message}")
    return float(res.x[0])


def uniform_prior_bias(logits: np.ndarray) -> np.ndarray:
    """Analytic class-bias under a uniform target prior.

    Returns ``b`` of shape (K,) such that ``z - b`` has the model's marginal
    prediction shifted toward uniform.

    Math
    ----
    p_calibrated(y|x) ∝ q(y|x) * p_target(y) / p_model(y)
    For uniform p_target = 1/K, the constant ``log(1/K)`` is absorbed into
    softmax, leaving ``log p_model(y)`` as the additive logit shift.

    ``p_model(y) = E_x[softmax(logits)_y]`` is estimated from the *calibration*
    set passed in, not the test set.
    """
    p = softmax(logits, axis=-1)
    # add tiny floor so log doesn't blow up if a class never gets >0 mass
    p_marginal = p.mean(axis=0).clip(min=1e-12)
    return np.log(p_marginal)


def fit_class_bias_l2(
    logits: np.ndarray,
    labels: np.ndarray,
    lam: float,
    init_b: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fit additive 12-D class bias by minimising NLL + lam * ||b||_2^2.

    Returns ``b`` of shape (K,); calibrated logits are ``z - b``.
    """
    K = logits.shape[1]
    if init_b is None:
        init_b = np.zeros(K, dtype=np.float64)
    N = len(labels)
    onehot = np.zeros((N, K), dtype=np.float64)
    onehot[np.arange(N), labels] = 1.0
    label_freq = onehot.mean(axis=0)

    def obj(b: np.ndarray) -> tuple[float, np.ndarray]:
        scaled = logits - b
        log_probs = scaled - log_sum_exp(scaled, axis=-1, keepdims=True)
        nll = -log_probs[np.arange(N), labels].mean()
        p = softmax(scaled, axis=-1)
        # d/db_j NLL = freq(j) - mean_n p[n,j]
        grad = label_freq - p.mean(axis=0) + 2.0 * lam * b
        return float(nll + lam * float((b * b).sum())), grad

    res = minimize(
        obj,
        x0=init_b.astype(np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"gtol": 1e-9, "ftol": 1e-12, "maxiter": 500},
    )
    if not res.success:
        warnings.warn(f"fit_class_bias_l2 L-BFGS-B did not converge (lam={lam}): {res.message}")
    return res.x.astype(np.float64)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@dataclass
class FoldOOF:
    """Per-fold equal-weight ensemble logits, labels, filenames, performers."""

    fold: int
    logits: np.ndarray  # (N_k, K)
    labels: np.ndarray  # (N_k,)
    filenames: list[str]
    performers: list[str]


def load_member_oof(
    artifacts_dir: Path,
    exp_name: str,
    num_folds: int = NUM_FOLDS,
) -> list[tuple[np.ndarray, np.ndarray, list[str]]]:
    """Load OOF (logits, labels, filenames) per fold for one ensemble member."""
    folds: list[tuple[np.ndarray, np.ndarray, list[str]]] = []
    for f in range(num_folds):
        d = artifacts_dir / exp_name / f"fold_{f:02d}"
        logits = np.load(d / "oof_logits.npy")
        labels = np.load(d / "oof_labels.npy")
        with open(d / "oof_filenames.txt") as fh:
            fnames = fh.read().strip().split("\n")
        if len(fnames) != logits.shape[0]:
            raise ValueError(
                f"{exp_name} fold {f:02d}: filename rows ({len(fnames)}) "
                f"!= logits rows ({logits.shape[0]})"
            )
        if labels.shape[0] != logits.shape[0]:
            raise ValueError(
                f"{exp_name} fold {f:02d}: labels rows ({labels.shape[0]}) "
                f"!= logits rows ({logits.shape[0]})"
            )
        folds.append((logits, labels, fnames))
    return folds


def load_ensemble_oof(
    artifacts_dir: Path,
    experiments: list[str],
    num_folds: int = NUM_FOLDS,
) -> list[FoldOOF]:
    """Load all members and return equal-weight averaged per-fold ensembles.

    Asserts label/filename order is byte-identical across members per fold.
    """
    members = [load_member_oof(artifacts_dir, e, num_folds=num_folds) for e in experiments]
    out: list[FoldOOF] = []
    for f in range(num_folds):
        ref_logits, ref_labels, ref_fnames = members[0][f]
        stack = [ref_logits]
        for mi, m in enumerate(members[1:], start=1):
            li, la, fn = m[f]
            if fn != ref_fnames:
                raise ValueError(
                    f"fold {f:02d}: filename order differs between "
                    f"{experiments[0]} and {experiments[mi]}"
                )
            if not np.array_equal(la, ref_labels):
                raise ValueError(
                    f"fold {f:02d}: labels differ between "
                    f"{experiments[0]} and {experiments[mi]}"
                )
            stack.append(li)
        ens = np.stack(stack, axis=0).mean(axis=0).astype(np.float64)
        performers = [parse_performer_id(s) for s in ref_fnames]
        out.append(FoldOOF(
            fold=f,
            logits=ens,
            labels=ref_labels.astype(np.int64),
            filenames=ref_fnames,
            performers=performers,
        ))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    return float(f1_score(
        labels, preds,
        average="macro",
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    ))


def per_class_f1(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.asarray(f1_score(
        labels, preds,
        average=None,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    ), dtype=np.float64)


# ---------------------------------------------------------------------------
# Inner CV for L2 lambda (Method C)
# ---------------------------------------------------------------------------

def inner_lambda_search(
    train_folds: list[FoldOOF],
    lambdas: list[float],
    rng: np.random.Generator,
) -> tuple[float, dict]:
    """Pick L2 lambda via inner LPO across the 9 outer-train folds.

    For each candidate lambda, compute the **sample-weighted** mean inner-val
    NLL over the n leave-one-fold-out splits of ``train_folds`` (LPO folds
    have non-uniform sizes — 4×864 + 6×756 — so unweighted averaging would
    bias toward small folds). Return the lambda with the lowest mean.
    """
    if not lambdas:
        raise ValueError("`lambdas` must contain at least one candidate")
    n_inner = len(train_folds)
    if n_inner < 2:
        raise ValueError(f"need >= 2 inner folds, got {n_inner}")
    # Per-(lambda, held) we store (nll, n_samples) so we can sample-weight.
    per_inner: dict[float, list[tuple[float, int]]] = {lam: [] for lam in lambdas}
    K = train_folds[0].logits.shape[1]

    for held in range(n_inner):
        # Build inner train (concatenate all but ``held``) and inner val (= held).
        inner_train_logits = np.concatenate(
            [tf.logits for i, tf in enumerate(train_folds) if i != held], axis=0,
        )
        inner_train_labels = np.concatenate(
            [tf.labels for i, tf in enumerate(train_folds) if i != held], axis=0,
        )
        inner_val_logits = train_folds[held].logits
        inner_val_labels = train_folds[held].labels
        n_val = len(inner_val_labels)

        for lam in lambdas:
            # Random init for L-BFGS-B (small noise; convex problem so solution
            # is essentially unique, but seeded init keeps things reproducible).
            init_b = 0.01 * rng.standard_normal(K)
            b = fit_class_bias_l2(
                inner_train_logits, inner_train_labels, lam=lam, init_b=init_b,
            )
            cal = inner_val_logits - b
            per_inner[lam].append((nll_loss(cal, inner_val_labels), n_val))

    # Sample-weighted mean NLL = sum_held(nll * n_val) / sum_held(n_val)
    mean_nll: dict[float, float] = {}
    for lam, pairs in per_inner.items():
        total_n = sum(n for _, n in pairs)
        weighted = sum(nll * n for nll, n in pairs)
        mean_nll[lam] = float(weighted / total_n)
    best_lam = min(mean_nll.items(), key=lambda kv: kv[1])[0]
    return best_lam, mean_nll


# ---------------------------------------------------------------------------
# Outer LPO calibration drivers
# ---------------------------------------------------------------------------

def calibrate_baseline(folds: list[FoldOOF]) -> list[np.ndarray]:
    """No-op calibration; returns raw logits per fold."""
    return [f.logits.copy() for f in folds]


def calibrate_temperature(
    folds: list[FoldOOF],
    seed: int,
) -> tuple[list[np.ndarray], dict]:
    """Outer LPO: fit T on the 9 training folds, apply to the held-out fold."""
    rng = np.random.default_rng(seed)
    cal: list[np.ndarray] = []
    fitted_T: list[float] = []
    for k in range(len(folds)):
        train_logits = np.concatenate(
            [folds[j].logits for j in range(len(folds)) if j != k], axis=0,
        )
        train_labels = np.concatenate(
            [folds[j].labels for j in range(len(folds)) if j != k], axis=0,
        )
        # init T near 1, with small noise per seed (convex; tiny init noise OK).
        init_T = float(1.0 + 0.05 * rng.standard_normal())
        init_T = max(init_T, 0.5)  # keep positive and sensible
        T = fit_temperature(train_logits, train_labels, init=init_T)
        fitted_T.append(T)
        cal.append(folds[k].logits / T)
    return cal, {"per_fold_T": fitted_T, "mean_T": float(np.mean(fitted_T))}


def calibrate_uniform_prior(
    folds: list[FoldOOF],
) -> tuple[list[np.ndarray], dict]:
    """Outer LPO: estimate p_model(y) on training folds, subtract log p_model from val logits."""
    cal: list[np.ndarray] = []
    fitted_b: list[np.ndarray] = []
    for k in range(len(folds)):
        train_logits = np.concatenate(
            [folds[j].logits for j in range(len(folds)) if j != k], axis=0,
        )
        b = uniform_prior_bias(train_logits)
        fitted_b.append(b)
        cal.append(folds[k].logits - b)
    return cal, {
        "per_fold_b": [list(map(float, b)) for b in fitted_b],
        "mean_b": list(map(float, np.mean(np.stack(fitted_b, axis=0), axis=0))),
    }


def calibrate_class_bias_l2(
    folds: list[FoldOOF],
    seed: int,
    lambdas: list[float],
) -> tuple[list[np.ndarray], dict]:
    """Outer LPO with inner 9-fold lambda selection per outer fold."""
    rng = np.random.default_rng(seed)
    K = folds[0].logits.shape[1]
    cal: list[np.ndarray] = []
    info_per_outer: list[dict] = []
    fitted_b: list[np.ndarray] = []

    for k in range(len(folds)):
        train_folds = [folds[j] for j in range(len(folds)) if j != k]
        best_lam, inner_nll_map = inner_lambda_search(train_folds, lambdas, rng)
        # Refit on the union of the 9 outer-train folds with the chosen lambda.
        train_logits = np.concatenate([tf.logits for tf in train_folds], axis=0)
        train_labels = np.concatenate([tf.labels for tf in train_folds], axis=0)
        init_b = 0.01 * rng.standard_normal(K)
        b = fit_class_bias_l2(train_logits, train_labels, lam=best_lam, init_b=init_b)
        fitted_b.append(b)
        cal.append(folds[k].logits - b)
        info_per_outer.append({
            "outer_fold": k,
            "best_lambda": float(best_lam),
            "inner_nll": {f"{lam:.4g}": v for lam, v in inner_nll_map.items()},
            "fitted_b": list(map(float, b)),
        })
    return cal, {
        "per_outer": info_per_outer,
        "mean_b": list(map(float, np.mean(np.stack(fitted_b, axis=0), axis=0))),
    }


# ---------------------------------------------------------------------------
# Direction-consistency check for class_bias_l2
# ---------------------------------------------------------------------------

def direction_consistency(
    per_fold_b: list[np.ndarray],
    full_b: np.ndarray,
    eps: float = 1e-3,
) -> dict:
    """Fraction of per-class bias signs that agree between nested fits and full fit.

    A class is counted as "agrees" if (a) full fit's bias and the per-fold
    fit's bias have the same sign, OR (b) both are within ``eps`` of zero
    (treated as effectively no shift). Comparison uses ``np.isclose`` so
    that ``-0.0 == 0.0`` is robust.
    """
    K = len(full_b)
    # Treat |b| < eps as "no shift" (sign 0); else use the actual sign.
    def _trinary_sign(arr: np.ndarray) -> np.ndarray:
        return np.where(np.abs(arr) < eps, 0.0, np.sign(arr))

    for i, b in enumerate(per_fold_b):
        if b.shape != (K,):
            raise ValueError(
                f"per_fold_b[{i}] has shape {b.shape}, expected ({K},)"
            )
    full_sign = _trinary_sign(full_b)
    agree_per_class = np.zeros(K, dtype=np.int64)
    total_per_class = np.zeros(K, dtype=np.int64)
    for b in per_fold_b:
        s = _trinary_sign(b)
        for c in range(K):
            total_per_class[c] += 1
            if np.isclose(s[c], full_sign[c]):
                agree_per_class[c] += 1
    per_class_rate = agree_per_class / np.maximum(total_per_class, 1)
    return {
        "per_class_agreement_rate": list(map(float, per_class_rate)),
        "overall_agreement_rate": float(per_class_rate.mean()),
        "num_classes_above_75pct": int((per_class_rate >= 0.75).sum()),
    }


# ---------------------------------------------------------------------------
# Sample-level paired bootstrap
# ---------------------------------------------------------------------------

def paired_bootstrap_f1_diff(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    labels: np.ndarray,
    n_iter: int = 1000,
    seed: int = 42,
) -> dict:
    """Sample-level bootstrap distribution of ``F1(preds_b) - F1(preds_a)``.

    Returns mean, std, and 95% CI of the diff. Positive => B beats A.

    Note: sample-level resampling treats each clip as independent. For
    leave-performer-out CV, the true unit of independence is the performer
    (clusters of 100+ clips), so this CI is **mildly optimistic** and should
    be reported as such. A cluster bootstrap (resampling performers) would
    be the strictly correct alternative; we follow plan §5.1's sample-level
    spec for comparability with prior phases.
    """
    if not (len(preds_a) == len(preds_b) == len(labels)):
        raise ValueError("preds_a, preds_b, labels must be same length")
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    N = len(labels)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, N, size=N)
        f_a = macro_f1(preds_a[idx], labels[idx])
        f_b = macro_f1(preds_b[idx], labels[idx])
        diffs[i] = f_b - f_a
    # ddof=1 std requires n_iter >= 2; fall back to 0.0 for the degenerate
    # smoke-test case (n_iter=1).
    std_pp = (
        float(diffs.std(ddof=1) * 100.0) if n_iter >= 2 else 0.0
    )
    return {
        "mean_diff_pp": float(diffs.mean() * 100.0),
        "std_diff_pp": std_pp,
        "ci_2.5_pp": float(np.quantile(diffs, 0.025) * 100.0),
        "ci_97.5_pp": float(np.quantile(diffs, 0.975) * 100.0),
        "n_iter": int(n_iter),
        "seed": int(seed),
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def evaluate_oof(cal_per_fold: list[np.ndarray], folds: list[FoldOOF]) -> dict:
    """Concatenate calibrated per-fold logits, compute aggregate + per-fold F1."""
    cal_all = np.concatenate(cal_per_fold, axis=0)
    labels_all = np.concatenate([f.labels for f in folds], axis=0)
    preds_all = cal_all.argmax(axis=1)
    nll = nll_loss(cal_all, labels_all)
    overall_f1 = macro_f1(preds_all, labels_all)
    overall_acc = float((preds_all == labels_all).mean())
    pcf1 = per_class_f1(preds_all, labels_all)
    per_fold_f1 = []
    for cf, ff in zip(cal_per_fold, folds):
        per_fold_f1.append(macro_f1(cf.argmax(axis=1), ff.labels))
    return {
        "overall_f1": float(overall_f1),
        "overall_acc": float(overall_acc),
        "overall_nll": float(nll),
        "per_fold_f1": list(map(float, per_fold_f1)),
        "per_fold_f1_mean": float(np.mean(per_fold_f1)),
        "per_fold_f1_std": float(np.std(per_fold_f1, ddof=1)),
        "per_class_f1": {IDX_TO_EMOTION[c]: float(pcf1[c]) for c in range(NUM_CLASSES)},
        "preds": preds_all,  # not JSON; will be popped before writing
        "labels": labels_all,
    }


def go_no_go_decision(
    method_metrics: dict[str, dict],
    bootstrap_results: dict[str, dict],
    direction_check: dict[str, dict],
    baseline_name: str = "baseline",
) -> dict:
    """Apply exp067 Go criteria (plan §5.1)."""
    base_f1 = method_metrics[baseline_name]["overall_f1_seed_mean"]
    base_pcf1 = method_metrics[baseline_name]["per_class_f1_seed_mean"]
    out: dict[str, dict] = {}
    for name, m in method_metrics.items():
        if name == baseline_name:
            continue
        delta = m["overall_f1_seed_mean"] - base_f1
        # criterion 1: +0.20pt
        c1 = delta >= 0.0020  # 0.20pt = 0.0020 in raw F1
        # criterion 2: 2+ priority classes improve
        improvements = []
        for cls in PRIORITY_CLASSES:
            improvements.append(m["per_class_f1_seed_mean"][cls] - base_pcf1[cls])
        c2 = sum(1 for d in improvements if d > 0) >= 2
        # criterion 3: bootstrap CI lower bound > -0.10pt
        ci_low = bootstrap_results[name]["ci_2.5_pp"]
        c3 = ci_low > -0.10
        # criterion 4 (only meaningful for class_bias_l2)
        c4 = True
        if name in direction_check:
            c4 = direction_check[name]["num_classes_above_75pct"] >= 9
        verdict = "GO" if (c1 and c2 and c3 and c4) else "NO-GO"
        out[name] = {
            "delta_f1_pp": float(delta * 100.0),
            "criterion_1_plus_0.20pt": bool(c1),
            "criterion_2_2plus_priority_improve": bool(c2),
            "priority_class_deltas_pp": {
                cls: float(d * 100.0) for cls, d in zip(PRIORITY_CLASSES, improvements)
            },
            "criterion_3_ci_lower_above_minus_0.10pp": bool(c3),
            "criterion_4_direction_consistency_75pct_9classes": bool(c4),
            "verdict": verdict,
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--experiments", multiple=True, default=DEFAULT_ENSEMBLE,
    help="Experiment folder names (default: production 7-way).",
)
@click.option("--num-folds", default=NUM_FOLDS, type=int)
@click.option("--num-seeds", default=3, type=int,
              help="Optimisation init seeds (T-only / L2 fits). >= 1.")
@click.option("--lambdas", default="0.001,0.01,0.1,1.0,10.0",
              help="Comma-separated L2 lambda candidates for class_bias_l2.")
@click.option("--bootstrap-iter", default=1000, type=int,
              help="Sample-level paired bootstrap iterations.")
@click.option("--bootstrap-seed", default=42, type=int)
@click.option("--out-dir", default="experiments/exp067_class_prior_calibration",
              type=click.Path(file_okay=False))
@click.option("--wandb/--no-wandb", default=False,
              help="Log a single summary run to wandb (project: acii2026-diema).")
@click.option(
    "--methods",
    multiple=True,
    default=("baseline", "temperature", "uniform_prior", "class_bias_l2"),
    help="Which calibration methods to run.",
)
def main(
    experiments: tuple,
    num_folds: int,
    num_seeds: int,
    lambdas: str,
    bootstrap_iter: int,
    bootstrap_seed: int,
    out_dir: str,
    wandb: bool,
    methods: tuple,
) -> None:
    env = EnvConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lam_list = [float(x) for x in lambdas.split(",") if x.strip()]
    methods = tuple(methods)
    if "baseline" not in methods:
        methods = ("baseline",) + methods

    if num_seeds < 1:
        raise click.BadParameter("--num-seeds must be >= 1")
    if bootstrap_iter < 1:
        raise click.BadParameter("--bootstrap-iter must be >= 1")
    if "class_bias_l2" in methods and not lam_list:
        raise click.BadParameter(
            "--lambdas must contain at least one positive candidate when "
            "class_bias_l2 is in --methods"
        )
    if any(lam < 0 for lam in lam_list):
        raise click.BadParameter("--lambdas entries must be non-negative")

    click.echo(f"Loading {len(experiments)}-way OOF...")
    folds = load_ensemble_oof(env.artifacts_dir, list(experiments), num_folds=num_folds)
    total_n = sum(f.logits.shape[0] for f in folds)
    click.echo(f"  Loaded {len(folds)} folds, total {total_n} OOF samples")
    for f in folds:
        click.echo(f"  fold {f.fold:02d}: N={f.logits.shape[0]}, "
                   f"performers={sorted(set(f.performers))}")

    # ------------------------------------------------------------------
    # Run each method; for stochastic methods (temperature, class_bias_l2)
    # repeat for ``num_seeds`` to capture optimisation jitter.
    # ------------------------------------------------------------------
    method_runs: dict[str, list[dict]] = {m: [] for m in methods}
    method_extra: dict[str, list[dict]] = {m: [] for m in methods}
    method_calibrated_logits: dict[str, list[list[np.ndarray]]] = {m: [] for m in methods}

    for method in methods:
        seeds_to_run = [0] if method in ("baseline", "uniform_prior") else list(range(num_seeds))
        for s in seeds_to_run:
            click.echo(f"\n[{method}] seed={s}")
            if method == "baseline":
                cal = calibrate_baseline(folds)
                extra = {}
            elif method == "temperature":
                cal, extra = calibrate_temperature(folds, seed=s)
                click.echo(f"  fitted T per outer fold: "
                           + ", ".join(f"{t:.3f}" for t in extra["per_fold_T"])
                           + f"  (mean={extra['mean_T']:.3f})")
            elif method == "uniform_prior":
                cal, extra = calibrate_uniform_prior(folds)
                click.echo(f"  mean b per class: "
                           + ", ".join(f"{IDX_TO_EMOTION[c]}={extra['mean_b'][c]:+.3f}"
                                       for c in range(NUM_CLASSES)))
            elif method == "class_bias_l2":
                cal, extra = calibrate_class_bias_l2(folds, seed=s, lambdas=lam_list)
                lams_picked = [info["best_lambda"] for info in extra["per_outer"]]
                click.echo(f"  picked lambdas per outer fold: {lams_picked}")
            else:
                raise click.UsageError(f"unknown method: {method}")
            metrics = evaluate_oof(cal, folds)
            click.echo(
                f"  overall F1 = {metrics['overall_f1']*100:.3f}%, "
                f"acc = {metrics['overall_acc']*100:.2f}%, NLL = {metrics['overall_nll']:.4f}, "
                f"per-fold F1 mean = {metrics['per_fold_f1_mean']*100:.3f}% "
                f"(std {metrics['per_fold_f1_std']*100:.3f})"
            )
            method_runs[method].append(metrics)
            method_extra[method].append(extra)
            method_calibrated_logits[method].append(cal)

    # ------------------------------------------------------------------
    # Aggregate seed-mean metrics per method.
    # ------------------------------------------------------------------
    method_metrics_summary: dict[str, dict] = {}
    for method, runs in method_runs.items():
        f1s = [r["overall_f1"] for r in runs]
        accs = [r["overall_acc"] for r in runs]
        nlls = [r["overall_nll"] for r in runs]
        per_fold_means = [r["per_fold_f1_mean"] for r in runs]
        # per-class F1: average across seeds
        pcf1_seed_mean = {}
        for cls in IDX_TO_EMOTION.values():
            pcf1_seed_mean[cls] = float(np.mean([r["per_class_f1"][cls] for r in runs]))
        method_metrics_summary[method] = {
            "num_seeds": len(runs),
            "overall_f1_seed_mean": float(np.mean(f1s)),
            "overall_f1_seed_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            "overall_acc_seed_mean": float(np.mean(accs)),
            "overall_nll_seed_mean": float(np.mean(nlls)),
            "per_fold_f1_seed_mean": float(np.mean(per_fold_means)),
            "per_class_f1_seed_mean": pcf1_seed_mean,
        }
        click.echo(
            f"\n[{method}] seed-mean F1 = {method_metrics_summary[method]['overall_f1_seed_mean']*100:.3f}% "
            f"(std {method_metrics_summary[method]['overall_f1_seed_std']*100:.3f}), "
            f"acc = {method_metrics_summary[method]['overall_acc_seed_mean']*100:.2f}%"
        )

    # ------------------------------------------------------------------
    # Sample-level paired bootstrap for each non-baseline method.
    # Use seed-0 calibrated predictions (deterministic methods) or
    # seed-mean of probability (averaging logits across seeds before argmax).
    # ------------------------------------------------------------------
    baseline_preds = method_runs["baseline"][0]["preds"]
    labels_all = method_runs["baseline"][0]["labels"]
    bootstrap_results: dict[str, dict] = {}
    for method in methods:
        if method == "baseline":
            continue
        # Average calibrated logits across seeds (more stable than picking seed 0).
        avg_logits = np.mean(
            np.stack(
                [np.concatenate(cal_seed, axis=0) for cal_seed in method_calibrated_logits[method]],
                axis=0,
            ),
            axis=0,
        )
        preds = avg_logits.argmax(axis=1)
        boot = paired_bootstrap_f1_diff(
            baseline_preds, preds, labels_all,
            n_iter=bootstrap_iter, seed=bootstrap_seed,
        )
        bootstrap_results[method] = boot
        click.echo(
            f"[{method}] bootstrap diff vs baseline: "
            f"mean={boot['mean_diff_pp']:+.3f}pp, "
            f"95%CI [{boot['ci_2.5_pp']:+.3f}, {boot['ci_97.5_pp']:+.3f}]"
        )

    # ------------------------------------------------------------------
    # Direction-consistency check (class_bias_l2 only).
    # ------------------------------------------------------------------
    direction_check: dict[str, dict] = {}
    if "class_bias_l2" in method_extra and len(method_extra["class_bias_l2"]) > 0:
        # Use seed-0 nested-LPO bias and a separately fitted full-data bias.
        per_fold_b = [
            np.asarray(info["fitted_b"])
            for info in method_extra["class_bias_l2"][0]["per_outer"]
        ]
        # full-data fit (no nesting), with lambda chosen by inner CV on the full set.
        all_logits = np.concatenate([f.logits for f in folds], axis=0)
        all_labels = np.concatenate([f.labels for f in folds], axis=0)
        rng = np.random.default_rng(0)
        # full-data lambda: do an "inner" CV using all 10 folds.
        full_lam, _ = inner_lambda_search(folds, lam_list, rng)
        full_b = fit_class_bias_l2(all_logits, all_labels, lam=full_lam, init_b=np.zeros(NUM_CLASSES))
        direction_check["class_bias_l2"] = direction_consistency(per_fold_b, full_b)
        direction_check["class_bias_l2"]["full_b"] = list(map(float, full_b))
        direction_check["class_bias_l2"]["full_lambda"] = float(full_lam)
        click.echo(
            f"[class_bias_l2] direction consistency: "
            f"overall {direction_check['class_bias_l2']['overall_agreement_rate']*100:.1f}%, "
            f"{direction_check['class_bias_l2']['num_classes_above_75pct']}/12 classes ≥75%"
        )

    # ------------------------------------------------------------------
    # Go/No-Go decision per criteria.
    # ------------------------------------------------------------------
    decision = go_no_go_decision(
        method_metrics_summary, bootstrap_results, direction_check,
    )
    for name, d in decision.items():
        click.echo(
            f"[{name}] verdict: {d['verdict']}  "
            f"(+{d['delta_f1_pp']:+.3f}pp, "
            f"priority improvements: {d['priority_class_deltas_pp']})"
        )

    # ------------------------------------------------------------------
    # Persist results.
    # ------------------------------------------------------------------
    # Strip preds/labels before JSON serialisation (numpy arrays).
    runs_for_json: dict[str, list[dict]] = {}
    for m, runs in method_runs.items():
        runs_for_json[m] = []
        for r in runs:
            r2 = {k: v for k, v in r.items() if k not in ("preds", "labels")}
            runs_for_json[m].append(r2)

    results = {
        "config": {
            "experiments": list(experiments),
            "num_folds": int(num_folds),
            "num_seeds": int(num_seeds),
            "lambdas": lam_list,
            "bootstrap_iter": int(bootstrap_iter),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "baseline_overall_f1": float(method_metrics_summary["baseline"]["overall_f1_seed_mean"]),
        "method_metrics": method_metrics_summary,
        "method_runs_per_seed": runs_for_json,
        "method_extra": method_extra,
        "bootstrap": bootstrap_results,
        "direction_check": direction_check,
        "decision": decision,
    }
    json_path = out / "results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, default=_json_default)
    click.echo(f"\nWrote {json_path}")

    # Per-class F1 quick view.
    pcf1_path = out / "per_class_f1.json"
    pcf1 = {m: method_metrics_summary[m]["per_class_f1_seed_mean"] for m in methods}
    with open(pcf1_path, "w") as fh:
        json.dump(pcf1, fh, indent=2)
    click.echo(f"Wrote {pcf1_path}")

    bs_path = out / "bootstrap.json"
    with open(bs_path, "w") as fh:
        json.dump(bootstrap_results, fh, indent=2)
    click.echo(f"Wrote {bs_path}")

    if wandb:
        try:
            import wandb as wb
        except ImportError:
            click.echo("[wandb] python package not installed; skipping logging", err=True)
        else:
            wb.init(
                project="acii2026-diema",
                name="exp067_class_prior_calibration",
                config=results["config"],
                tags=["exp067", "phase3AL", "calibration"],
            )
            for m, ms in method_metrics_summary.items():
                wb.summary[f"{m}/overall_f1"] = ms["overall_f1_seed_mean"]
                wb.summary[f"{m}/overall_acc"] = ms["overall_acc_seed_mean"]
                wb.summary[f"{m}/overall_nll"] = ms["overall_nll_seed_mean"]
                for cls, v in ms["per_class_f1_seed_mean"].items():
                    wb.summary[f"{m}/per_class_f1/{cls}"] = v
            for m, b in bootstrap_results.items():
                wb.summary[f"{m}/bootstrap_mean_diff_pp"] = b["mean_diff_pp"]
                wb.summary[f"{m}/bootstrap_ci_lower_pp"] = b["ci_2.5_pp"]
                wb.summary[f"{m}/bootstrap_ci_upper_pp"] = b["ci_97.5_pp"]
            for m, d in decision.items():
                wb.summary[f"{m}/verdict"] = d["verdict"]
                wb.summary[f"{m}/delta_f1_pp"] = d["delta_f1_pp"]
            wb.finish()
            click.echo("[wandb] logged summary run")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


if __name__ == "__main__":
    main()
