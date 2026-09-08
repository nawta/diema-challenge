"""Ensemble weight search + LPO-CV robustness for DIEM-A OOF predictions.

Implements the six basic strategies (equal / F1-weighted / F1^2 / LOO-drop /
greedy / softmax_mean) plus:

  * Nelder-Mead continuous global-weight optimization (argmax F1 objective,
    30 random restarts),
  * Per-class weights (M x 12 parameters) via Nelder-Mead,
  * Performer-LPO 5-fold CV to detect overfitting (random split is NOT honest
    because OOF rows carry performer identity; we group by the
    ``(country, actor_id)`` pair derived from ``train_data.csv``),
  * Paired bootstrap CI on the OOF delta between optimized and equal weights.

All search objectives use the same macro-F1 definition as training
(``torchmetrics.F1Score(task="multiclass", average="macro")``) via the
helper :func:`_per_class_f1` from :mod:`tools.ensemble_models`.

Usage::

    python tools/ensemble_weight_search.py \
      --experiments exp002_a04_smooth \
      --experiments exp003_ctrgcn_a00 \
      ...

See also: ``tools/ensemble_models.py`` (single-ensemble eval).
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.env import EnvConfig
from tools.ensemble_models import load_experiment_oof, _per_class_f1, _softmax


# ---------- metric helpers ----------

def _macro_f1(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = logits.argmax(axis=1)
    return float(_per_class_f1(preds, labels).mean())


def _acc(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = logits.argmax(axis=1)
    return float((preds == labels).mean())


def _combine(aligned_logits: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    w = weights.astype(np.float32)
    w = w / (w.sum() + 1e-12)
    combined = np.zeros_like(aligned_logits[0])
    for wi, L in zip(w, aligned_logits):
        combined += wi * L
    return combined


# ---------- alignment with strong asserts ----------

def _align(loaded: list):
    """Align logits / labels by filename order of the first experiment.

    Asserts every experiment has exactly the same filename set (no extras,
    no duplicates) so silent drops cannot go unnoticed (reviewer bug report,
    2026-04-17).
    """
    ref_fnames = loaded[0][2]
    ref_labels = loaded[0][1]
    ref_set = set(ref_fnames)
    if len(ref_set) != len(ref_fnames):
        raise ValueError("first experiment has duplicate filenames")

    aligned = [loaded[0][0]]
    for i, (logits, labels, filenames) in enumerate(loaded[1:], start=1):
        if len(filenames) != len(ref_fnames):
            raise ValueError(
                f"experiment {i}: filename count {len(filenames)} != "
                f"ref {len(ref_fnames)}"
            )
        if set(filenames) != ref_set:
            extra = set(filenames) - ref_set
            missing = ref_set - set(filenames)
            raise ValueError(
                f"experiment {i}: set mismatch, extra={len(extra)}, "
                f"missing={len(missing)}"
            )
        fn_to_row = {fn: j for j, fn in enumerate(filenames)}
        row_order = [fn_to_row[fn] for fn in ref_fnames]
        aligned.append(logits[row_order])
        if not np.array_equal(labels[row_order], ref_labels):
            raise ValueError(f"experiment {i}: label mismatch after align")
    return aligned, ref_labels, ref_fnames


# ---------- performer-LPO grouping ----------

def _load_performer_groups(ref_fnames: list[str]) -> tuple[np.ndarray, int]:
    """Return an ``(N,)`` int array assigning each filename to a performer
    id. Performer id = index into the sorted list of ``(country, actor_id)``
    pairs in ``train_data.csv``.

    Reviewer-2026-04-17 fix: previously unknown filenames were silently
    remapped to group ``0`` (a real performer), which contaminated LPO.
    Now each unknown filename is assigned its own pseudo-performer id
    (``num_perf + i``) so it is kept isolated in its own fold.
    """
    from diema.training.dual_head_trainer import build_filename_to_performer_index
    train_csv = Path("data/diema_challenge/raw/train_data.csv")
    if not train_csv.exists():
        raise FileNotFoundError(f"{train_csv} not found; cannot build LPO groups")
    fn_map, num_perf = build_filename_to_performer_index(train_csv)
    groups = np.empty(len(ref_fnames), dtype=np.int64)
    next_unknown = num_perf
    for i, fn in enumerate(ref_fnames):
        if fn in fn_map:
            groups[i] = fn_map[fn]
        else:
            groups[i] = next_unknown
            next_unknown += 1
    n_unknown = next_unknown - num_perf
    if n_unknown > 0:
        click.echo(
            f"  [warn] {n_unknown} filenames had no performer mapping; "
            f"each assigned a distinct pseudo-performer id "
            f"(total groups: {next_unknown})",
            err=True,
        )
    return groups, next_unknown


# ---------- optimizers ----------

def _opt_global_nm(A_fit: np.ndarray, y_fit: np.ndarray, n_restart: int = 30,
                  seed: int = 42, return_dispersion: bool = False):
    """Nelder-Mead global-weight search. Returns ``(weights, train_f1)``.

    If ``return_dispersion=True``, also returns the array of final F1s across
    all restarts (reviewer 2026-04-17: restart-selection artifact check).
    """
    from scipy.optimize import minimize
    M = A_fit.shape[0]

    def neg_f1(w):
        w = np.abs(w)
        w = w / (w.sum() + 1e-12)
        combined = (w[:, None, None] * A_fit).sum(0)
        preds = combined.argmax(1)
        return -_per_class_f1(preds, y_fit).mean()

    w0 = np.ones(M) / M
    best_f1 = -neg_f1(w0)
    best_w = w0.copy()
    all_f1s = [best_f1]
    rng = np.random.default_rng(seed)
    for s in range(n_restart):
        w_start = w0.copy() if s == 0 else rng.dirichlet(np.ones(M) * 5)
        res = minimize(
            neg_f1, w_start, method="Nelder-Mead",
            options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 500},
        )
        f1 = -res.fun
        all_f1s.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            w = np.abs(res.x)
            best_w = w / (w.sum() + 1e-12)
    if return_dispersion:
        return best_w, best_f1, np.array(all_f1s)
    return best_w, best_f1


def _opt_per_class_nm(A_fit: np.ndarray, y_fit: np.ndarray,
                     w0_global: np.ndarray, n_restart: int = 30,
                     seed: int = 42) -> tuple[np.ndarray, float]:
    """Per-class Nelder-Mead: ``W[m, c]``, ``M * C = 84`` params for M=7/C=12.
    """
    from scipy.optimize import minimize
    M, N, C = A_fit.shape

    def neg_f1(W_flat):
        W = np.abs(W_flat.reshape(M, C))
        W = W / (W.sum(0, keepdims=True) + 1e-12)
        combined = (W[:, None, :] * A_fit).sum(0)
        preds = combined.argmax(1)
        return -_per_class_f1(preds, y_fit).mean()

    W0 = np.tile(w0_global[:, None], (1, C))
    best_f1 = -neg_f1(W0.flatten())
    best_W = W0.copy()
    rng = np.random.default_rng(seed)
    for s in range(n_restart):
        W_start = W0.flatten() if s == 0 else (
            W0.flatten() + rng.normal(0, 0.05, M * C)
        )
        res = minimize(
            neg_f1, W_start, method="Nelder-Mead",
            options={"xatol": 5e-4, "fatol": 1e-6, "maxiter": 5000},
        )
        f1 = -res.fun
        if f1 > best_f1:
            best_f1 = f1
            W = np.abs(res.x.reshape(M, C))
            W = W / (W.sum(0, keepdims=True) + 1e-12)
            best_W = W
    return best_W, best_f1


def _greedy(A_fit: np.ndarray, y_fit: np.ndarray, max_steps: int = 20) -> np.ndarray:
    """Caruana forward greedy with bincount-weighted model accumulation."""
    M = A_fit.shape[0]
    ind_f1 = np.array(
        [_per_class_f1(A_fit[m].argmax(1), y_fit).mean() for m in range(M)]
    )
    selected = [int(ind_f1.argmax())]
    for step in range(max_steps):
        best_add = (-1.0, None)
        for cand in range(M):
            trial = selected + [cand]
            w = np.bincount(trial, minlength=M).astype(np.float32)
            combined = (w[:, None, None] * A_fit).sum(0)
            preds = combined.argmax(1)
            f1 = _per_class_f1(preds, y_fit).mean()
            if f1 > best_add[0]:
                best_add = (f1, cand)
        cand = best_add[1]
        prev_w = np.bincount(selected, minlength=M).astype(np.float32)
        prev_f1 = _per_class_f1(
            (prev_w[:, None, None] * A_fit).sum(0).argmax(1), y_fit,
        ).mean()
        if best_add[0] <= prev_f1 + 1e-5 and step > 0:
            break
        selected.append(cand)
    w_final = np.bincount(selected, minlength=M).astype(np.float32)
    return w_final / (w_final.sum() + 1e-12)


# ---------- CV (LPO-structured + random for comparison) ----------

def _make_folds_lpo(groups: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """Assign each unique performer group to one of k folds (LPO-structured).
    Returns list of length k, each entry is an array of sample indices.
    """
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    perf_to_fold = {p: int(i % k) for i, p in enumerate(uniq)}
    fold_masks = [np.zeros(groups.shape[0], dtype=bool) for _ in range(k)]
    for i, g in enumerate(groups):
        fold_masks[perf_to_fold[g]][i] = True
    return [np.where(m)[0] for m in fold_masks]


def _make_folds_random(N: int, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    return np.array_split(perm, k)


def _eval_weight_strategy(A: np.ndarray, labels: np.ndarray, folds: list[np.ndarray],
                         strategy: str, base_seed: int = 42,
                         return_preds: bool = False) -> dict:
    """Return per-fold train/val F1 (+ optionally val predictions/labels).

    Reviewer-2026-04-17 fix: seed propagation per fold. Previously all folds
    used NM with ``seed=42`` → identical Dirichlet restart directions. Now
    each fold gets ``base_seed + fi`` so restarts are independent.
    """
    M = A.shape[0]
    train_f1s, val_f1s = [], []
    all_val_preds = [] if return_preds else None
    all_val_labels = [] if return_preds else None
    all_val_preds_equal = [] if return_preds else None
    for fi, val_idx in enumerate(folds):
        val_mask = np.zeros(A.shape[1], dtype=bool)
        val_mask[val_idx] = True
        train_idx = np.where(~val_mask)[0]
        A_fit = A[:, train_idx, :]
        A_val = A[:, val_idx, :]
        y_fit = labels[train_idx]
        y_val = labels[val_idx]
        fold_seed = base_seed + fi

        if strategy == "equal":
            w = np.ones(M) / M
            fit_preds = (w[:, None, None] * A_fit).sum(0).argmax(1)
            val_preds = (w[:, None, None] * A_val).sum(0).argmax(1)
            train_f1 = _per_class_f1(fit_preds, y_fit).mean()
            val_f1 = _per_class_f1(val_preds, y_val).mean()
        elif strategy == "greedy":
            w = _greedy(A_fit, y_fit)
            fit_preds = (w[:, None, None] * A_fit).sum(0).argmax(1)
            val_preds = (w[:, None, None] * A_val).sum(0).argmax(1)
            train_f1 = _per_class_f1(fit_preds, y_fit).mean()
            val_f1 = _per_class_f1(val_preds, y_val).mean()
        elif strategy == "nm_global":
            w, train_f1 = _opt_global_nm(A_fit, y_fit, n_restart=20, seed=fold_seed)
            val_preds = (w[:, None, None] * A_val).sum(0).argmax(1)
            val_f1 = _per_class_f1(val_preds, y_val).mean()
        elif strategy == "nm_perclass":
            w_init, _ = _opt_global_nm(A_fit, y_fit, n_restart=10, seed=fold_seed)
            W, train_f1 = _opt_per_class_nm(
                A_fit, y_fit, w_init, n_restart=10, seed=fold_seed + 10_000,
            )
            val_preds = (W[:, None, :] * A_val).sum(0).argmax(1)
            val_f1 = _per_class_f1(val_preds, y_val).mean()
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        train_f1s.append(float(train_f1))
        val_f1s.append(float(val_f1))
        if return_preds:
            all_val_preds.append(val_preds)
            all_val_labels.append(y_val)

    out = {"train_f1": train_f1s, "val_f1": val_f1s}
    if return_preds:
        out["val_preds"] = np.concatenate(all_val_preds)
        out["val_labels"] = np.concatenate(all_val_labels)
    return out


def _paired_sample_bootstrap_delta(
    val_preds_a: np.ndarray, val_preds_b: np.ndarray, val_labels: np.ndarray,
    B: int = 2000, seed: int = 42,
) -> tuple[float, float, float]:
    """Paired sample-level bootstrap of Macro-F1 delta (method a - method b).

    Each bootstrap resample draws N samples with replacement from the paired
    (preds_a[i], preds_b[i], label[i]) tuples, recomputes Macro-F1 for each,
    and takes the difference. Returns (mean_delta, ci_low, ci_high) at 95%.

    Much tighter than fold-level bootstrap (k=5 → only 126 distinct multisets).
    """
    N = val_labels.shape[0]
    rng = np.random.default_rng(seed)
    deltas = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        f1a = _per_class_f1(val_preds_a[idx], val_labels[idx]).mean()
        f1b = _per_class_f1(val_preds_b[idx], val_labels[idx]).mean()
        deltas[b] = f1a - f1b
    mean = float(deltas.mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return mean, float(lo), float(hi)


# ---------- CLI ----------

@click.command()
@click.option("--experiments", required=True, multiple=True,
              help="Experiment folder names.")
@click.option("--cv-mode", default="both",
              type=click.Choice(["lpo", "random", "both", "none"]),
              help="CV split mode (lpo = leave-performer-out).")
@click.option("--cv-k", default=10, type=int,
              help="CV fold count (default 10 to match training LPO).")
@click.option("--cv-seed", default=42, type=int, help="CV split seed.")
@click.option("--cv-seed-repeats", default=1, type=int,
              help="Repeat CV with multiple partition seeds and aggregate.")
@click.option("--skip-nm", is_flag=True, default=False,
              help="Skip Nelder-Mead (faster).")
@click.option("--bootstrap-b", default=2000, type=int,
              help="Sample-level bootstrap iterations.")
def main(experiments: tuple, cv_mode: str, cv_k: int, cv_seed: int,
         cv_seed_repeats: int, skip_nm: bool, bootstrap_b: int):
    env = EnvConfig()
    exp_list = list(experiments)

    # -- load + align
    loaded = []
    for exp in exp_list:
        exp_dir = env.artifacts_dir / exp
        logits, labels, filenames = load_experiment_oof(exp_dir)
        click.echo(f"  {exp}: {logits.shape}")
        loaded.append((logits, labels, filenames))

    aligned, labels, ref_fnames = _align(loaded)
    A = np.stack(aligned, axis=0)  # (M, N, C)
    M, N, C = A.shape

    # -- individual F1
    per_model_f1 = np.array([_macro_f1(A[m], labels) for m in range(M)])
    click.echo("\n=== Individual ===")
    for name, f1 in zip(exp_list, per_model_f1):
        click.echo(f"  {name}: F1 {f1 * 100:.2f}%")

    # -- 6 basic strategies on full OOF (for reference; these can overfit)
    click.echo("\n=== OOF full (no holdout - exploration only) ===")
    w_eq = np.ones(M) / M
    click.echo(
        f"  equal              : F1 {_macro_f1(_combine(aligned, w_eq), labels) * 100:.2f}%  "
        f"Acc {_acc(_combine(aligned, w_eq), labels) * 100:.2f}%"
    )
    for tag, w_raw in [
        ("F1-weighted       ", per_model_f1),
        ("F1-weighted (x^2) ", per_model_f1 ** 2),
    ]:
        w = w_raw / w_raw.sum()
        combined = _combine(aligned, w)
        click.echo(
            f"  {tag}: F1 {_macro_f1(combined, labels) * 100:.2f}%  "
            f"Acc {_acc(combined, labels) * 100:.2f}%"
        )
    w_greedy = _greedy(A, labels)
    click.echo(
        f"  greedy (Caruana)   : F1 {_macro_f1(_combine(aligned, w_greedy), labels) * 100:.2f}%  "
        f"Acc {_acc(_combine(aligned, w_greedy), labels) * 100:.2f}%  "
        f"w={w_greedy.round(3).tolist()}"
    )
    if not skip_nm:
        w_nm, _ = _opt_global_nm(A, labels, n_restart=30)
        click.echo(
            f"  NM global          : F1 {_macro_f1(_combine(aligned, w_nm), labels) * 100:.2f}%  "
            f"Acc {_acc(_combine(aligned, w_nm), labels) * 100:.2f}%  "
            f"w={w_nm.round(3).tolist()}"
        )
        W_pc, _ = _opt_per_class_nm(A, labels, w_nm, n_restart=30)
        combined_pc = (W_pc[:, None, :] * A).sum(0)
        preds_pc = combined_pc.argmax(1)
        click.echo(
            f"  NM per-class       : F1 {_per_class_f1(preds_pc, labels).mean() * 100:.2f}%  "
            f"Acc {(preds_pc == labels).mean() * 100:.2f}%  "
            f"(M*C={M * C} params)"
        )

    # -- CV robustness check
    cv_modes = []
    if cv_mode in ("lpo", "both"):
        cv_modes.append("lpo")
    if cv_mode in ("random", "both"):
        cv_modes.append("random")

    if cv_modes:
        groups = None
        if "lpo" in cv_modes:
            try:
                groups, _ = _load_performer_groups(ref_fnames)
            except FileNotFoundError as e:
                click.echo(f"  [warn] LPO CV unavailable: {e}", err=True)
                cv_modes.remove("lpo")

    for mode in cv_modes:
        click.echo(
            f"\n=== {cv_k}-fold {mode.upper()} CV"
            f" ({cv_seed_repeats} partition seed{'s' if cv_seed_repeats > 1 else ''}) ==="
        )

        strategies = ["equal", "greedy"]
        if not skip_nm:
            strategies += ["nm_global", "nm_perclass"]

        # Aggregate over partition seeds
        all_train = {s: [] for s in strategies}
        all_val = {s: [] for s in strategies}
        all_preds = {s: [] for s in strategies}
        all_labels = []

        for rep in range(cv_seed_repeats):
            seed = cv_seed + rep * 1000
            if mode == "lpo":
                folds = _make_folds_lpo(groups, cv_k, seed)
            else:
                folds = _make_folds_random(N, cv_k, seed)
            if rep == 0:
                if mode == "lpo":
                    uniq = [np.unique(groups[f]) for f in folds]
                    click.echo(
                        f"  seed 0: {len(np.unique(groups))} performers, "
                        f"fold sizes={[len(f) for f in folds]}, "
                        f"unique perf/fold={[len(u) for u in uniq]}"
                    )
                else:
                    click.echo(
                        f"  seed 0: fold sizes={[len(f) for f in folds]}"
                    )

            for strat in strategies:
                out = _eval_weight_strategy(
                    A, labels, folds, strat, base_seed=seed,
                    return_preds=True,
                )
                all_train[strat].extend(out["train_f1"])
                all_val[strat].extend(out["val_f1"])
                all_preds[strat].append(out["val_preds"])
                if strat == strategies[0]:
                    all_labels.append(out["val_labels"])

        # Aggregate and report
        click.echo(f"  {'strategy':<14}  {'train F1':>10}  {'val F1':>10}  {'val vs equal':>14}  {'fold SE':>9}")
        eq_val = np.array(all_val["equal"])
        for strat in strategies:
            tr = np.array(all_train[strat])
            vl = np.array(all_val[strat])
            delta = (vl - eq_val).mean() * 100
            se = vl.std(ddof=1) / np.sqrt(len(vl)) * 100 if len(vl) > 1 else 0.0
            click.echo(
                f"  {strat:<14}  {tr.mean() * 100:>9.2f}%  {vl.mean() * 100:>9.2f}%  "
                f"{delta:>+13.2f}pt  {se:>8.2f}pt"
            )

        # Per-fold table (first seed only)
        n_per_seed = cv_k
        click.echo(f"  per-fold val F1 (seed 0):")
        header = "    fold " + " ".join(f"{s:>11}" for s in strategies)
        click.echo(header)
        for fi in range(n_per_seed):
            row = f"    {fi:>4} " + " ".join(
                f"{all_val[s][fi] * 100:>10.2f}%" for s in strategies
            )
            click.echo(row)

        # Robust statistics for greedy (reviewer flag: mean dominated by 1 fold)
        greedy_deltas = (np.array(all_val["greedy"]) - eq_val) * 100
        click.echo(
            f"  greedy per-fold deltas: "
            f"mean={greedy_deltas.mean():+.2f}pt, "
            f"median={float(np.median(greedy_deltas)):+.2f}pt, "
            f"trimmed_mean={float(np.mean(np.sort(greedy_deltas)[1:-1])):+.2f}pt"
        )

        # Sample-level paired bootstrap on concatenated val predictions
        if "nm_global" in strategies and bootstrap_b > 0:
            val_labels_full = np.concatenate(all_labels)
            eq_preds_full = np.concatenate(all_preds["equal"])
            nm_preds_full = np.concatenate(all_preds["nm_global"])
            pc_preds_full = (
                np.concatenate(all_preds["nm_perclass"])
                if "nm_perclass" in all_preds and all_preds["nm_perclass"]
                else None
            )
            gr_preds_full = np.concatenate(all_preds["greedy"])

            for a_name, a_preds, b_name, b_preds in [
                ("NM global", nm_preds_full, "equal", eq_preds_full),
                ("NM per-class", pc_preds_full, "equal", eq_preds_full) if pc_preds_full is not None else (None, None, None, None),
                ("greedy", gr_preds_full, "equal", eq_preds_full),
            ]:
                if a_name is None:
                    continue
                mean_d, lo, hi = _paired_sample_bootstrap_delta(
                    a_preds, b_preds, val_labels_full, B=bootstrap_b, seed=cv_seed,
                )
                sig = "(signif)" if (lo > 0 or hi < 0) else "(n.s.)"
                click.echo(
                    f"  {a_name:>12} vs {b_name:<6} sample-level bootstrap 95% CI: "
                    f"[{lo * 100:+.2f}, {hi * 100:+.2f}]pt  "
                    f"(mean {mean_d * 100:+.2f}pt) {sig}"
                )


if __name__ == "__main__":
    main()
