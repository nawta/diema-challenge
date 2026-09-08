""" logit-scale normalisation ablation on the 11-way ensemble.

Related:
- experiments/exp090_ensemble_patterns/sweep.py (original 7-way fusion sweep)
- experiments/exp091_ensemble_lma_faithfulness/performer_block_bootstrap.py
    (helpers for assembling the 11-way OOF logits)

A review flagged that the published +1.15 pp gain of raw-logit-
mean over probability-mean (paper Table 1 / Fig.2 lift path) could be a
*logit-scale* artefact — a single high-norm member dominating the
average. This script tests that hypothesis by comparing five fusion
conventions on the 11-way ensemble:

  1. Raw logit-mean (canonical: ``softmax(log(softmax(L)).mean(0))``;
     equivalent to ``softmax(L_raw.mean(0))`` when each L has the same
     scale — which they don't in general)
  2. Probability mean (``softmax(L).mean(0)``)
  3. Temperature-scaled logit-mean — fit per-member temperature T_i on
     fold-internal val so each member is roughly calibrated, then mean
  4. Per-model z-normalised logit-mean — z-score each member's logits
     across (samples, classes) before averaging
  5. Borda / rank averaging — rank classes per sample per member, average
     ranks; argmax of summed ranks

All conventions are applied to the same OOF logits (per-fold).

The headline output is per-convention Macro-F1 (pooled and per-fold mean
± SD) and per-class F1 deltas vs the canonical convention.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.env import EnvConfig  # noqa: E402

from experiments.exp091_ensemble_lma_faithfulness.performer_block_bootstrap import (  # noqa: E402
    PROD_MEMBERS, EXTERNAL_OOFS, _softmax,
)


NUM_CLASS = 12


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.asarray(
        f1_score(y_true, y_pred, average=None,
                 labels=list(range(NUM_CLASS)), zero_division=0)
    )


def _build_11way_member_logits(env: EnvConfig) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str],
]:
    """Concatenate per-fold PROD + aligned external OOF logits.

    Returns ``(member_logits, labels, fold_ids, filenames, member_names)``.
    ``member_logits.shape == (11, N, 12)``.
    """
    art = env.artifacts_dir
    # PROD
    fns_chunks: list[np.ndarray] = []
    lbls_chunks: list[np.ndarray] = []
    fold_id_chunks: list[np.ndarray] = []
    prod_logits = {m: [] for m in PROD_MEMBERS}
    for fid in range(10):
        fns = np.loadtxt(art / "exp002_a04_smooth" / f"fold_{fid:02d}" / "oof_filenames.txt",
                         dtype=str)
        lbls = np.load(art / "exp002_a04_smooth" / f"fold_{fid:02d}" / "oof_labels.npy")
        fns_chunks.append(fns)
        lbls_chunks.append(lbls)
        fold_id_chunks.append(np.full(len(fns), fid, dtype=np.int64))
        for m in PROD_MEMBERS:
            prod_logits[m].append(
                np.load(art / m / f"fold_{fid:02d}" / "oof_logits.npy")
            )
    filenames = np.concatenate(fns_chunks)
    labels = np.concatenate(lbls_chunks)
    fold_ids = np.concatenate(fold_id_chunks)
    for m in PROD_MEMBERS:
        prod_logits[m] = np.concatenate(prod_logits[m], axis=0)
    # External, aligned to PROD order
    ext_logits: dict[str, np.ndarray] = {}
    for name, (lg_p, fn_p) in EXTERNAL_OOFS.items():
        L = np.load(env.project_root / lg_p)
        F = np.load(env.project_root / fn_p, allow_pickle=True)
        idx_map = {str(f): i for i, f in enumerate(F)}
        ordered = np.array([idx_map[str(f)] for f in filenames])
        ext_logits[name] = L[ordered]
    member_names: list[str] = list(PROD_MEMBERS) + list(EXTERNAL_OOFS.keys())
    member_logits = np.stack(
        [prod_logits[m] for m in PROD_MEMBERS]
        + [ext_logits[m] for m in EXTERNAL_OOFS.keys()],
        axis=0,
    )
    return member_logits, labels, fold_ids, filenames, member_names


# --------------------------------------------------------------------------- #
# Fusion conventions
# --------------------------------------------------------------------------- #


def fuse_logit_mean(member_logits: np.ndarray) -> np.ndarray:
    """Canonical: softmax of mean log-prob (= raw-logit-mean convention)."""
    P = _softmax(member_logits)
    return _softmax(np.log(np.clip(P, 1e-12, 1.0)).mean(axis=0))


def fuse_probability_mean(member_logits: np.ndarray) -> np.ndarray:
    """Plain probability mean (arithmetic average of softmax)."""
    return _softmax(member_logits).mean(axis=0)


def fuse_temperature_scaled_logit_mean(
    member_logits: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    Ts_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Per-member temperature fit on a held-out 50/50 split of each val
    fold's OOF (to avoid cheating on the eval samples), then logit-mean.

    For each member, we fit a single scalar T_i that minimises the val
    NLL on the calibration half; we then apply softmax(L_i / T_i) to
    the eval half.
    """
    if Ts_grid is None:
        Ts_grid = np.linspace(0.5, 3.0, 26)  # 0.5, 0.6, ..., 3.0
    M, N, C = member_logits.shape
    Ts_per_member_per_fold = np.ones((M, 10), dtype=np.float64)
    eval_mask = np.zeros(N, dtype=bool)

    rng = np.random.default_rng(42)
    for fid in range(10):
        fold_idx = np.where(fold_ids == fid)[0]
        rng.shuffle(fold_idx)
        half = len(fold_idx) // 2
        calib_idx = fold_idx[:half]
        evals_idx = fold_idx[half:]
        eval_mask[evals_idx] = True
        y_calib = labels[calib_idx]
        for m in range(M):
            best_T, best_nll = 1.0, np.inf
            for T in Ts_grid:
                p = _softmax(member_logits[m, calib_idx] / T)
                nll = -np.log(np.clip(p[np.arange(len(y_calib)), y_calib],
                                      1e-12, 1.0)).mean()
                if nll < best_nll:
                    best_nll, best_T = nll, float(T)
            Ts_per_member_per_fold[m, fid] = best_T

    fused = np.zeros((N, C), dtype=np.float64)
    for fid in range(10):
        idx = np.where(fold_ids == fid)[0]
        Ls = member_logits[:, idx, :]  # (M, n_fold, C)
        scaled = np.stack([Ls[m] / Ts_per_member_per_fold[m, fid]
                           for m in range(M)], axis=0)
        P = _softmax(scaled)
        fused[idx] = _softmax(np.log(np.clip(P, 1e-12, 1.0)).mean(axis=0))
    return fused, eval_mask, Ts_per_member_per_fold


def fuse_zscore_logit_mean(member_logits: np.ndarray) -> np.ndarray:
    """Per-member z-score over (sample, class), then logit-mean."""
    M = member_logits.shape[0]
    z = np.zeros_like(member_logits, dtype=np.float64)
    for m in range(M):
        flat = member_logits[m].reshape(-1)
        mu, sd = float(flat.mean()), float(flat.std() + 1e-12)
        z[m] = (member_logits[m] - mu) / sd
    P = _softmax(z)
    return _softmax(np.log(np.clip(P, 1e-12, 1.0)).mean(axis=0))


def fuse_rank_borda(member_logits: np.ndarray) -> np.ndarray:
    """Borda-like rank averaging. For each (sample, member), rank classes
    by logit (0 = lowest, C-1 = highest), average ranks across members,
    then return softmax-like normalisation of summed ranks (for downstream
    argmax this is monotone equivalent to mean rank).
    """
    M, N, C = member_logits.shape
    ranks = np.zeros((M, N, C), dtype=np.float64)
    for m in range(M):
        # argsort gives positions of sorted values; argsort again gives ranks.
        ranks[m] = member_logits[m].argsort(axis=-1).argsort(axis=-1)
    mean_rank = ranks.mean(axis=0)  # (N, C)
    return _softmax(mean_rank)  # softmax for downstream consistency


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out-json",
                   default="experiments/exp091_ensemble_lma_faithfulness/r3_results.json")
    args = p.parse_args()

    env = EnvConfig()
    print("[R3] Loading 11-way member OOF logits …")
    t0 = time.time()
    member_logits, labels, fold_ids, filenames, member_names = _build_11way_member_logits(env)
    print(f"[R3] member_logits {member_logits.shape}, "
          f"n_samples {len(labels)} ({time.time()-t0:.1f}s)")
    # Member raw-logit norm (rough proxy for scale dominance) — diagnose
    # whether any single member's logits dwarf the others' before fusion.
    print("\n[R3] Per-member logit std (raw):")
    for i, name in enumerate(member_names):
        sd = float(member_logits[i].std())
        print(f"  {name:>40s}: std={sd:.3f}")

    # Helper: per-convention metrics on a given sample subset.
    def _metrics(pred: np.ndarray, mask: np.ndarray) -> dict:
        y = labels[mask]
        p = pred[mask]
        f1_pooled = _macro_f1(y, p)
        f1_per_fold = np.array([
            _macro_f1(labels[(fold_ids == f) & mask],
                      pred[(fold_ids == f) & mask])
            for f in range(10)
        ])
        return {
            "pooled_f1": float(f1_pooled),
            "per_fold_mean": float(f1_per_fold.mean()),
            "per_fold_std": float(f1_per_fold.std(ddof=1)),
            "per_class_f1": _per_class_f1(y, p).tolist(),
        }

    # Pre-compute the temperature-scaled fusion + eval_mask so we can
    # *report all conventions on the same eval_mask* (post-C2 fix,
    # 2026-06-12). The temperature is fit on calib_half only; we evaluate
    # every fusion convention on eval_half, so the comparison is
    # apples-to-apples.
    fused_T, eval_mask, Ts = fuse_temperature_scaled_logit_mean(
        member_logits, labels, fold_ids,
    )
    full_mask = np.ones(len(labels), dtype=bool)

    # Build the predictions per convention.
    fused_funcs = {
        "raw_logit_mean":      fuse_logit_mean(member_logits),
        "probability_mean":    fuse_probability_mean(member_logits),
        "zscore_logit_mean":   fuse_zscore_logit_mean(member_logits),
        "rank_borda":          fuse_rank_borda(member_logits),
        "temperature_scaled":  fused_T,
    }

    results = {}
    print("\n[R3] Per-convention Macro-F1 on FULL OOF + EVAL_HALF "
          "(both reported for transparency; eval_half is the fair-comparison"
          " set vs temperature_scaled):")
    for name, fused in fused_funcs.items():
        pred = fused.argmax(1)
        m_full = _metrics(pred, full_mask)
        m_eval = _metrics(pred, eval_mask)
        results[name] = {
            "full": m_full,
            "eval_half": m_eval,
        }
        print(f"  {name:>22s}: full {m_full['pooled_f1']*100:.2f} %, "
              f"eval-half {m_eval['pooled_f1']*100:.2f} %")

    results["temperature_scaled"]["temperatures_per_member_per_fold"] = Ts.tolist()
    results["temperature_scaled"]["temperature_mean"] = float(Ts.mean())
    results["temperature_scaled"]["temperature_std"] = float(Ts.std())

    # Summary headline 1: full-data Δ vs canonical raw_logit_mean (the
    # comparison used by 4 of 5 conventions; matches paper Fig.2 lift).
    canon_full = results["raw_logit_mean"]["full"]["pooled_f1"]
    print("\n[R3] full-OOF Δ vs canonical raw_logit_mean:")
    for k, r in results.items():
        if k == "raw_logit_mean":
            continue
        v = r["full"]["pooled_f1"]
        print(f"  {k:>22s}: Δ = {(v - canon_full)*100:+.2f} pp "
              f"(pooled F1 {v*100:.2f} % vs {canon_full*100:.2f} %)")

    # Summary headline 2: eval-half Δ vs canonical raw_logit_mean (the
    # fair comparison that includes temperature_scaled).
    canon_eval = results["raw_logit_mean"]["eval_half"]["pooled_f1"]
    print("\n[R3] eval-half Δ vs canonical raw_logit_mean (POST-C2-FIX, "
          "fair comparison incl. temperature_scaled):")
    for k, r in results.items():
        if k == "raw_logit_mean":
            continue
        v = r["eval_half"]["pooled_f1"]
        print(f"  {k:>22s}: Δ = {(v - canon_eval)*100:+.2f} pp "
              f"(pooled F1 {v*100:.2f} % vs {canon_eval*100:.2f} %)")

    out = {
        "method": "logit_scale_ablation",
        "ensemble": "11way",
        "n_samples": int(len(labels)),
        "members": list(member_names),
        "results": results,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\n[R3] saved {args.out_json}")


if __name__ == "__main__":
    main()
