""" refresh the single-model LMA Spearman ρ (paper Table 3 P11)
with the n + bootstrap CI + permutation null statistics demanded by the
a review.

Related:
- experiments/exp087_lma_attr_aux/eval_faithfulness.py  (original computation,
    publishes ρ = +0.500 without CI / null / explicit n)
- experiments/exp091_ensemble_lma_faithfulness/eval_ensemble_lma.py
    (this is the ensemble-side counterpart; we reuse its statistics machinery)
- experiments/exp078_temporal_spatial_saliency/results.npz
    (12 × 25 CTV spatial saliency from the Region-Aware single model)
- output/lma_attrs_rule.npz (32-D LMA attributes per clip)

Recomputes the published single-model LMA ρ using identical region groupings
and adds:
  - explicit n: per-emotion ρ on n=4 regions, then mean across 12 emotions
  - 95% bootstrap CI (emotion block, 1000 boot, seed=42)
  - permutation null (region permutation per emotion, 1000 iter, seed=42)
  - one-sided p-value: P(null ≥ observed)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.exp091_ensemble_lma_faithfulness.eval_ensemble_lma import (
    LMA_REGION,
    REGIONS,
    ORDER,
    per_emotion_zscore,
    lma_region_importance,
    per_emotion_spearman,
    permutation_null,
    bootstrap_ci_emotion_block,
)


# CTV-25 (virtual_root + BVH-24) regions, identical to exp087.
CTV_REGION = {
    "torso": [1, 2, 3, 4],
    "head":  [5, 6, 7, 8],
    "arms":  [9, 10, 11, 12, 13, 14, 15, 16],
    "legs":  [17, 18, 19, 20, 21, 22, 23, 24],
}


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--lma-npz", default="output/lma_attrs_rule.npz")
    p.add_argument("--saliency-npz",
                   default="experiments/exp078_temporal_spatial_saliency/results.npz")
    p.add_argument("--out-json",
                   default="experiments/exp091_ensemble_lma_faithfulness/results_single_model.json")
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    lma = np.load(args.lma_npz, allow_pickle=True)
    feats, labels = lma["features"], lma["labels"]
    Z_lma = per_emotion_zscore(feats, labels)
    lma_reg = lma_region_importance(Z_lma)
    print(f"[R5] LMA region importance shape: {lma_reg.shape}")

    sal_npz = np.load(args.saliency_npz, allow_pickle=True)
    spatial_sal = sal_npz["spatial_sal"]  # (12, 25)
    deep_reg = np.zeros((12, 4), dtype=np.float64)
    for ri, r in enumerate(REGIONS):
        deep_reg[:, ri] = spatial_sal[:, CTV_REGION[r]].mean(1)
    print(f"[R5] Single-model spatial-saliency region importance shape: {deep_reg.shape}")

    rho_per_emotion = per_emotion_spearman(lma_reg, deep_reg)
    rho_mean = float(rho_per_emotion.mean())

    # Bootstrap CI over emotions (n=12 paired ρ).
    ci_lo, ci_hi = bootstrap_ci_emotion_block(
        lma_reg, deep_reg, n_boot=args.n_boot, seed=args.seed
    )

    # Permutation null.
    nulls = permutation_null(lma_reg, deep_reg, n_perm=args.n_perm, seed=args.seed)
    null_mean = float(nulls.mean())
    null_std = float(nulls.std(ddof=1))
    # (count + 1) / (n + 1) Monte Carlo correction (Phipson & Smyth 2010),
    # avoiding the p=0 floor under finite-sample permutation testing.
    p_value = float((1.0 + np.sum(nulls >= rho_mean)) / (len(nulls) + 1.0))

    print()
    print("=" * 70)
    print("Single-model LMA faithfulness (R5; refreshes paper Table 3 P11)")
    print("=" * 70)
    print(f"  Spearman ρ:                       {rho_mean:+.4f}")
    print(f"  95% bootstrap CI (emotion block): [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  Permutation null mean / std:      {null_mean:+.4f} / {null_std:.4f}")
    print(f"  P(null ≥ observed):               {p_value:.4f}")
    print(f"  n_emotions: 12 ; n_regions: 4 ; n_total_pairs: 48")
    print()
    print("Per-emotion ρ (single-model spatial saliency × LMA):")
    for c, e in enumerate(ORDER):
        print(f"  {e:>10s}: {rho_per_emotion[c]:+.3f}")

    out = {
        "method": "single_model_spatial_saliency_vs_lma",
        "model": "exp034_regionaware_convtr_a00 (Region-Aware) via exp078 saliency",
        "spearman_mean": rho_mean,
        "spearman_ci95": [ci_lo, ci_hi],
        "permutation_null": {
            "n_perm": args.n_perm,
            "mean": null_mean,
            "std": null_std,
            "p_value": p_value,
            "seed": args.seed,
        },
        "bootstrap_ci": {
            "n_boot": args.n_boot,
            "lo": ci_lo,
            "hi": ci_hi,
            "block": "emotion",
            "n": 12,
            "seed": args.seed,
        },
        "per_emotion_spearman": {
            e: float(rho_per_emotion[c]) for c, e in enumerate(ORDER)
        },
        "n_emotions": 12,
        "n_regions": 4,
        "n_total_pairs": 48,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\n[R5] saved {args.out_json}")


if __name__ == "__main__":
    main()
