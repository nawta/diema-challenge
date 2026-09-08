""" performer-block + fold-level paired bootstrap CI on the
11-way vs STGCN++ baseline Macro-F1 delta — refresh paper Table II.

Related:
- experiments/exp090_ensemble_patterns/build_logitmean_submission.py
    (canonical 11-way logit-mean fusion)
- paper/reconcile_f1.py (canonical 11-way OOF Macro-F1 = 36.94 % pooled,
    36.80 ± 4.00 % per-fold mean ± SD; STGCN++ reproduced baseline
    = exp002_a04_smooth at 25.73 ± 4.03 % per-fold)

A review flagged that the published 95% CI on Table II was
computed by *sample-level* paired bootstrap on pooled OOF, which is
optimistic because LPO clips are nested in performers (each performer
contributes 108 clips, so the effective independent unit is the
performer, not the clip). This script reports three CIs to flank the
published interval:

  1. Sample-level paired bootstrap on pooled OOF (matches the current paper)
  2. **Performer-block paired bootstrap** on pooled OOF (resample 74
     performers with replacement; all 108 clips of each sampled performer
     come along)
  3. Fold-level paired bootstrap on the per-fold Macro-F1 delta
     (resample 10 folds with replacement; recompute mean delta)

All three are paired (per resample we compute Macro-F1 for both models
on the same resample, then take the delta), so they isolate the
estimator's variability rather than the noise floor of either model.

Inputs (pre-computed by ``_prepare_oof()``):
- 11-way logit-mean fused log-probs (from per-member OOFs)
- exp002_a04_smooth baseline OOF predictions (paper canonical STGCN++)
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.env import EnvConfig  # noqa: E402


PROD_MEMBERS: tuple[str, ...] = (
    "exp002_a04_smooth", "exp003_ctrgcn_a00", "exp004_skateformer_a01",
    "exp006_protogcn_a00", "exp020_conv1d_transformer_a01",
    "exp023_keypoint_pool_mlp_a00", "exp034_regionaware_convtr_a00",
)

EXTERNAL_OOFS: dict[str, tuple[str, str]] = {
    "motionbert": (
        "output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_logits.npy",
        "output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_filenames.npy",
    ),
    "c3d":     (
        "output/predictions/c3d_contact_only/oof_logits.npy",
        "output/predictions/c3d_contact_only/oof_filenames.npy",
    ),
    "mamp60":  (
        "output/predictions/mamp_ntu60xsub/oof_logits.npy",
        "output/predictions/mamp_ntu60xsub/oof_filenames.npy",
    ),
    "mamp120": (
        "output/predictions/mamp_ntu120xset/oof_logits.npy",
        "output/predictions/mamp_ntu120xset/oof_filenames.npy",
    ),
}

NUM_CLASS = 12


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _prepare_oof(env: EnvConfig, baseline_exp: str = "exp002_a04_smooth") -> dict:
    """Concatenate per-fold OOF, align externals, fuse 11-way.

    Returns a dict with:
        ``filenames`` (N,) str
        ``labels``    (N,)
        ``fold_ids``  (N,)
        ``pred_11way``(N,) int — argmax of logit-mean fusion
        ``pred_base`` (N,) int — argmax of baseline's per-fold OOF
        ``performer_ids`` (N,) str — e.g. "JP_06"
    """
    art = env.artifacts_dir
    out_pred = env.project_root / "output/predictions"

    # 1. PROD per-fold concat (filenames + labels + per-member logits)
    fns_chunks, lbls_chunks, fold_id_chunks = [], [], []
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
    # Baseline = whichever exp the user names. If it isn't already a PROD
    # member (e.g. exp001 for the official STGCN++ reproduction), load
    # its per-fold OOF separately. Filename order must match PROD; we
    # cross-check on a sample of fold_00.
    if baseline_exp in prod_logits:
        pred_base = prod_logits[baseline_exp].argmax(1)
    else:
        base_logits_chunks = []
        for fid in range(10):
            base_fns = np.loadtxt(
                art / baseline_exp / f"fold_{fid:02d}" / "oof_filenames.txt",
                dtype=str,
            )
            ref_fns = fns_chunks[fid]
            if not np.array_equal(base_fns, ref_fns):
                raise RuntimeError(
                    f"baseline ({baseline_exp}) fold_{fid:02d} filename "
                    f"order disagrees with PROD reference"
                )
            base_logits_chunks.append(
                np.load(art / baseline_exp / f"fold_{fid:02d}" / "oof_logits.npy")
            )
        pred_base = np.concatenate(base_logits_chunks, axis=0).argmax(1)

    # 2. External OOFs aligned to PROD filename order
    ext_logits: dict[str, np.ndarray] = {}
    for name, (lg_p, fn_p) in EXTERNAL_OOFS.items():
        L = np.load(env.project_root / lg_p)
        F = np.load(env.project_root / fn_p, allow_pickle=True)
        idx_map = {str(f): i for i, f in enumerate(F)}
        ordered = np.array([idx_map[str(f)] for f in filenames])
        ext_logits[name] = L[ordered]

    # 3. 11-way logit-mean fusion
    P_list = [_softmax(prod_logits[m]) for m in PROD_MEMBERS]
    for name in ext_logits:
        P_list.append(_softmax(ext_logits[name]))
    P = np.stack(P_list, axis=0)  # (11, N, 12)
    fused = _softmax(np.log(np.clip(P, 1e-12, 1.0)).mean(axis=0))
    pred_11way = fused.argmax(1)

    # 4. Performer IDs from filename prefix (e.g. "JP_06_...")
    pid_re = re.compile(r"^([A-Z]{2}_[0-9]+)_")
    performer_ids = np.array([
        pid_re.match(str(fn)).group(1) for fn in filenames
    ])

    return {
        "filenames": filenames,
        "labels": labels,
        "fold_ids": fold_ids,
        "pred_11way": pred_11way,
        "pred_base": pred_base,
        "performer_ids": performer_ids,
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def sample_level_paired_bootstrap(
    labels: np.ndarray,
    pred_a: np.ndarray, pred_b: np.ndarray,
    n_boot: int = 1000, seed: int = 42,
) -> tuple[float, float]:
    """Sample-level paired bootstrap on pooled OOF. Returns (ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    N = len(labels)
    deltas = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx = rng.integers(0, N, size=N)
        deltas[k] = _macro_f1(labels[idx], pred_a[idx]) - _macro_f1(labels[idx], pred_b[idx])
    return tuple(np.percentile(deltas, [2.5, 97.5]).tolist())


def performer_block_paired_bootstrap(
    labels: np.ndarray,
    pred_a: np.ndarray, pred_b: np.ndarray,
    performer_ids: np.ndarray,
    n_boot: int = 1000, seed: int = 42,
) -> tuple[float, float]:
    """Performer-block paired bootstrap. Resample performers with replacement;
    take all of each sampled performer's clips. Returns (ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    unique_perfs = np.unique(performer_ids)
    perf_to_idx: dict[str, np.ndarray] = {
        p: np.where(performer_ids == p)[0] for p in unique_perfs
    }
    K = len(unique_perfs)
    deltas = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        sampled_perfs = rng.choice(unique_perfs, size=K, replace=True)
        sampled_idx = np.concatenate([perf_to_idx[p] for p in sampled_perfs])
        deltas[k] = (_macro_f1(labels[sampled_idx], pred_a[sampled_idx])
                     - _macro_f1(labels[sampled_idx], pred_b[sampled_idx]))
    return tuple(np.percentile(deltas, [2.5, 97.5]).tolist())


def fold_level_paired_bootstrap(
    labels: np.ndarray,
    pred_a: np.ndarray, pred_b: np.ndarray,
    fold_ids: np.ndarray,
    n_boot: int = 1000, seed: int = 42,
) -> tuple[float, float]:
    """Fold-level paired bootstrap. Resample folds with replacement; recompute
    mean per-fold delta. Returns (ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    unique_folds = np.unique(fold_ids)
    per_fold_delta = np.empty(len(unique_folds), dtype=np.float64)
    for i, f in enumerate(unique_folds):
        m = fold_ids == f
        per_fold_delta[i] = (_macro_f1(labels[m], pred_a[m])
                             - _macro_f1(labels[m], pred_b[m]))
    F = len(per_fold_delta)
    deltas = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx = rng.integers(0, F, size=F)
        deltas[k] = per_fold_delta[idx].mean()
    return tuple(np.percentile(deltas, [2.5, 97.5]).tolist())


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="exp002_a04_smooth")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json",
                   default="experiments/exp091_ensemble_lma_faithfulness/r2_results.json")
    args = p.parse_args()

    env = EnvConfig()
    print(f"[R2] Preparing 11-way + baseline OOF (baseline={args.baseline}) …")
    t0 = time.time()
    oof = _prepare_oof(env, baseline_exp=args.baseline)
    print(f"[R2] Pooled n_samples={len(oof['labels'])}, "
          f"unique performers={len(set(oof['performer_ids']))}, "
          f"({time.time()-t0:.1f}s)")

    # Headline F1s
    pooled_f1_11way = _macro_f1(oof["labels"], oof["pred_11way"])
    pooled_f1_base = _macro_f1(oof["labels"], oof["pred_base"])
    pooled_delta = pooled_f1_11way - pooled_f1_base
    per_fold_f1_11way = np.array([
        _macro_f1(oof["labels"][oof["fold_ids"] == f],
                  oof["pred_11way"][oof["fold_ids"] == f])
        for f in range(10)
    ])
    per_fold_f1_base = np.array([
        _macro_f1(oof["labels"][oof["fold_ids"] == f],
                  oof["pred_base"][oof["fold_ids"] == f])
        for f in range(10)
    ])
    per_fold_delta = per_fold_f1_11way - per_fold_f1_base

    print()
    print("=" * 70)
    print("Headline Macro-F1 (pooled OOF)")
    print("=" * 70)
    print(f"  11-way (logit-mean):  {pooled_f1_11way*100:.4f} %  (paper canonical 36.94 %)")
    print(f"  STGCN++ baseline:     {pooled_f1_base*100:.4f} %  (paper canonical 25.73 %)")
    print(f"  Pooled Δ:             {pooled_delta*100:+.4f} pp")
    print()
    print("Per-fold paired Δ:")
    for fid in range(10):
        print(f"  fold_{fid:02d}: {per_fold_delta[fid]*100:+.4f} pp")
    print(f"  mean: {per_fold_delta.mean()*100:+.4f} ± {per_fold_delta.std(ddof=1)*100:.4f} pp")
    print(f"  sign test: {(per_fold_delta > 0).sum()}/{len(per_fold_delta)} folds positive")

    print()
    print("=" * 70)
    print("Paired bootstrap CIs (1000 iter, seed=42)")
    print("=" * 70)
    t0 = time.time()
    ci_sample = sample_level_paired_bootstrap(
        oof["labels"], oof["pred_11way"], oof["pred_base"],
        n_boot=args.n_boot, seed=args.seed,
    )
    print(f"  [sample-level on pooled OOF]   Δ CI95 = "
          f"[{ci_sample[0]*100:+.4f}, {ci_sample[1]*100:+.4f}] pp  "
          f"({time.time()-t0:.1f}s)")

    t0 = time.time()
    ci_perf = performer_block_paired_bootstrap(
        oof["labels"], oof["pred_11way"], oof["pred_base"],
        oof["performer_ids"], n_boot=args.n_boot, seed=args.seed,
    )
    print(f"  [performer-block]              Δ CI95 = "
          f"[{ci_perf[0]*100:+.4f}, {ci_perf[1]*100:+.4f}] pp  "
          f"({time.time()-t0:.1f}s)")

    t0 = time.time()
    ci_fold = fold_level_paired_bootstrap(
        oof["labels"], oof["pred_11way"], oof["pred_base"],
        oof["fold_ids"], n_boot=args.n_boot, seed=args.seed,
    )
    print(f"  [fold-level on per-fold Δ]     Δ CI95 = "
          f"[{ci_fold[0]*100:+.4f}, {ci_fold[1]*100:+.4f}] pp  "
          f"({time.time()-t0:.1f}s)")

    out = {
        "method": "paired_bootstrap_three_blockings",
        "ensemble": "11way_logitmean",
        "baseline_exp": args.baseline,
        "n_samples": int(len(oof["labels"])),
        "n_performers": int(len(set(oof["performer_ids"]))),
        "pooled_f1_11way": pooled_f1_11way,
        "pooled_f1_base": pooled_f1_base,
        "pooled_delta": pooled_delta,
        "per_fold_delta_mean": float(per_fold_delta.mean()),
        "per_fold_delta_std": float(per_fold_delta.std(ddof=1)),
        "per_fold_delta_paired_sign_positive_n": int((per_fold_delta > 0).sum()),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "ci95_sample_level": list(ci_sample),
        "ci95_performer_block": list(ci_perf),
        "ci95_fold_level_on_mean_delta": list(ci_fold),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\n[R2] saved {args.out_json}")


if __name__ == "__main__":
    main()
