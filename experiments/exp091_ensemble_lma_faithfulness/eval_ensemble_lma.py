""" ensemble-level LMA faithfulness via Spearman ρ.

Related:
- experiments/exp087_lma_attr_aux/eval_faithfulness.py
    Single-model version using exp078 spatial saliency. Computes per-emotion
    Spearman ρ between LMA region importance and deep saliency region.
    This script mirrors the protocol but replaces ``spatial_sal`` with
    *ensemble* per-emotion part-masking F1 drop.
- tools/explain_ensemble_part_masking.py
    Produces per-fold, per-(emotion, part) F1 drops we consume here.
- diema/features/lma_attributes.py
    32-D LMA names (Body / Effort / Shape / Space × 8 each).

Method
------

For the submitted ensemble (7-way PROD logit-mean), we computed the F1 drop
when each body part's joints were shuffled (permutation-importance). The
drop per (emotion, part) is interpreted as the *ensemble's reliance on
that part for that emotion*.

We then:

1. Aggregate the 6 DIEMA body parts → 4 LMA-comparable regions:
   ``head``, ``arms`` (= r_arm + l_arm), ``legs`` (= r_leg + l_leg),
   ``torso``. Sum the F1 drops within each region group (additive on
   independent parts).
2. For each emotion, compute the (4,) region-importance vector from the
   ensemble (mean across 10 folds, so n=10 paired observations per cell).
3. Compute the per-emotion (4,) LMA region z-score importance from
   ``output/lma_attrs_rule.npz`` (the same 32-D rule-based attributes
   the paper Table 3 LMA row uses).
4. Spearman ρ per emotion between the two (4,) vectors → mean = ensemble
   LMA faithfulness Spearman.
5. Compare to exp079 classical 5-feature baseline (+0.033) and the
   exp087 single-model spatial saliency result (+0.500).

Additional statistics (review-defense gate)
-------------------------------------------

* **n is explicit**: we compute Spearman over (12 emotions) × (4 regions)
  paired vectors → "per emotion ρ on n=4 regions" then mean over 12.
* **Permutation null**: shuffle the LMA attribute → region mapping 1000×
  with seed=42; mean ρ under null gives a p-value.
* **Bootstrap CI**: 1000 performer-block resample to derive 95% CI on
  the mean ρ.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from diema.features.lma_attributes import LMA_NAMES


# --------------------------------------------------------------------------- #
# Constants — kept compatible with exp087/eval_faithfulness.py
# --------------------------------------------------------------------------- #


# LMA feature → region (mirrors exp087 LMA_REGION exactly).
LMA_REGION: dict[str, list[str]] = {
    "head":  ["body.head_bow", "body.head_lateral_tilt", "shape.head_height",
              "shape.rise_sink"],
    "arms":  ["body.arm_openness", "body.lr_asym_speed", "body.lr_asym_pos",
              "shape.shoulder_width", "shape.spread_change", "shape.enclose_proxy",
              "body.shoulder_drop"],
    "legs":  ["space.locomotion_ratio", "space.root_xy_disp", "space.root_z_disp",
              "space.root_sway_xy_rms", "space.directness", "space.path_curvature",
              "space.cumulative_turn", "space.dominant_direction",
              "space.advance_recede" if "space.advance_recede" in LMA_NAMES
              else "shape.advance_recede"],
    "torso": ["body.trunk_lean", "shape.body_volume", "shape.contraction_change",
              "body.stillness_ratio", "effort.suddenness", "effort.sustainedness",
              "effort.strong_proxy", "effort.light_proxy", "effort.bound_proxy",
              "effort.free_proxy", "effort.intensity_peak", "effort.intensity_var"],
}
REGIONS = ["head", "arms", "legs", "torso"]
ORDER = ["anger", "contempt", "disgust", "fear", "joy", "sadness",
         "surprise", "jealousy", "shame", "guilt", "gratitude", "pride"]

# DIEMA 6-part → 4-region mapping (sum drops within a region group).
PART_TO_REGION: dict[str, str] = {
    "head": "head", "torso": "torso",
    "r_arm": "arms", "l_arm": "arms",
    "r_leg": "legs", "l_leg": "legs",
}


# --------------------------------------------------------------------------- #
# LMA region importance (identical to exp087)
# --------------------------------------------------------------------------- #


def per_emotion_zscore(feats: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """(N, D), (N,) → (12, D) per-emotion z-score vs global mean/std."""
    mu, sd = feats.mean(0), feats.std(0) + 1e-8
    Z = np.zeros((12, feats.shape[1]), dtype=np.float64)
    for c in range(12):
        m = labels == c
        Z[c] = (feats[m].mean(0) - mu) / sd
    return Z


def lma_region_importance(Z: np.ndarray) -> np.ndarray:
    """(12, 32) z-score → (12, 4) |z|-mean per region (head/arms/legs/torso)."""
    name_idx = {n: i for i, n in enumerate(LMA_NAMES)}
    out = np.zeros((12, 4), dtype=np.float64)
    for ri, r in enumerate(REGIONS):
        idx = [name_idx[n] for n in LMA_REGION[r] if n in name_idx]
        out[:, ri] = np.abs(Z[:, idx]).mean(1)
    return out


# --------------------------------------------------------------------------- #
# Ensemble per-(emotion, region) importance from part-masking CSVs
# --------------------------------------------------------------------------- #


def ensemble_region_importance(csv_path: Path) -> np.ndarray:
    """Read the aggregated CSV from explain_ensemble_part_masking.py and
    convert to (12 emotions, 4 regions) importance = mean F1 drop across folds,
    summed within each region group.

    The CSV has rows ``(ensemble, mask_mode, fold, country, intensity, part,
    baseline_f1, masked_f1, f1_drop, f1_drop_<emotion>×12)``. We use the
    ``f1_drop_<emotion>`` columns directly so the result is the *per-class*
    drop, not the macro-F1 drop.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    # Mean across folds per (part, emotion).
    drop_cols = [c for c in df.columns if c.startswith("f1_drop_")]
    agg = df.groupby("part")[drop_cols].mean()  # (6, 12)
    # Verify all 12 emotions present in expected order.
    emo_cols = [f"f1_drop_{e}" for e in ORDER]
    assert all(c in drop_cols for c in emo_cols), (
        f"missing emotion column; have {drop_cols}, expected {emo_cols}"
    )
    parts = agg.index.tolist()
    # Region importance per emotion: sum drops over parts in the region.
    region_imp = np.zeros((12, 4), dtype=np.float64)
    for ri, r in enumerate(REGIONS):
        parts_in_region = [p for p in parts if PART_TO_REGION[p] == r]
        for ci, e in enumerate(ORDER):
            region_imp[ci, ri] = agg.loc[parts_in_region, f"f1_drop_{e}"].sum()
    return region_imp


# --------------------------------------------------------------------------- #
# Permutation null + bootstrap CI
# --------------------------------------------------------------------------- #


def per_emotion_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-emotion (12,) Spearman ρ between (12, 4) and (12, 4) region vectors."""
    rho = np.zeros(12, dtype=np.float64)
    for c in range(12):
        r = spearmanr(a[c], b[c]).correlation
        rho[c] = 0.0 if np.isnan(r) else r
    return rho


def permutation_null(
    lma_reg: np.ndarray, deep_reg: np.ndarray, n_perm: int = 1000, seed: int = 42
) -> np.ndarray:
    """Shuffle region columns of ``deep_reg`` per emotion to break the
    LMA-vs-deep alignment, then recompute mean Spearman. The null mean
    should be ≈ 0 (random region permutation destroys correspondence).
    """
    rng = np.random.default_rng(seed)
    nulls = np.zeros(n_perm, dtype=np.float64)
    for k in range(n_perm):
        shuffled = np.empty_like(deep_reg)
        for c in range(12):
            perm = rng.permutation(4)
            shuffled[c] = deep_reg[c, perm]
        nulls[k] = float(np.mean(per_emotion_spearman(lma_reg, shuffled)))
    return nulls


def bootstrap_ci_emotion_block(
    lma_reg: np.ndarray, deep_reg: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> tuple[float, float]:
    """Emotion-block paired bootstrap on the per-emotion ρ vector.

    With only 12 emotions this is the natural block for "would the
    conclusion hold if the emotion set were slightly different?" — not
    quite a performer block (we don't have per-emotion-per-performer ρ
    here) but the right unit at this aggregation level. The exp087
    headline likewise reports an emotion-block mean.
    """
    rho = per_emotion_spearman(lma_reg, deep_reg)
    rng = np.random.default_rng(seed)
    boots = np.zeros(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx = rng.choice(12, size=12, replace=True)
        boots[k] = float(rho[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ensemble-csv",
        default="docs/analysis/ensemble_part_masking/ensemble_7way_part_importance_shuffle.csv",
        help="Aggregated CSV from explain_ensemble_part_masking.py",
    )
    p.add_argument("--lma-npz", default="output/lma_attrs_rule.npz")
    p.add_argument("--exp079-npz",
                   default="experiments/exp079_classical_kinematic_features/results.npz")
    p.add_argument("--out-json",
                   default="experiments/exp091_ensemble_lma_faithfulness/results.json")
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # ---- 1. LMA region importance ----
    lma = np.load(args.lma_npz, allow_pickle=True)
    feats, labels = lma["features"], lma["labels"]
    Z_lma = per_emotion_zscore(feats, labels)
    lma_reg = lma_region_importance(Z_lma)
    print(f"[exp091] LMA region importance shape: {lma_reg.shape}")

    # ---- 2. Ensemble region importance ----
    ens_reg = ensemble_region_importance(Path(args.ensemble_csv))
    print(f"[exp091] Ensemble region importance shape: {ens_reg.shape}")
    print("  per-emotion sum of region drops (sanity ~baseline_f1 - all_masked):")
    for c, e in enumerate(ORDER):
        print(f"    {e:>10s}: {ens_reg[c].sum():.4f}")

    # ---- 3. Per-emotion Spearman: ensemble vs LMA ----
    rho_ens = per_emotion_spearman(lma_reg, ens_reg)
    rho_ens_mean = float(rho_ens.mean())

    # ---- 4. Classical baseline from exp079 (BVH-24 5 features × 4 regions) ----
    s79 = np.load(args.exp079_npz, allow_pickle=True)
    z79 = s79["feat_zscore_per_feature"]  # (12, 24, 5)
    # BVH-24 region indices from exp087.
    BVH_REGION = {
        "torso": [0, 1, 2, 3], "head": [4, 5, 6, 7],
        "arms": [8, 9, 10, 11, 12, 13, 14, 15],
        "legs": [16, 17, 18, 19, 20, 21, 22, 23],
    }
    base_reg = np.zeros((12, 4), dtype=np.float64)
    for ri, r in enumerate(REGIONS):
        base_reg[:, ri] = np.abs(z79[:, BVH_REGION[r], :]).mean(axis=(1, 2))
    rho_base = per_emotion_spearman(lma_reg, base_reg)
    rho_base_mean = float(rho_base.mean())

    # ---- 5. Permutation null ----
    nulls = permutation_null(lma_reg, ens_reg, n_perm=args.n_perm, seed=args.seed)
    null_mean = float(nulls.mean())
    null_std = float(nulls.std(ddof=1))
    # p-value (one-sided): standard Monte Carlo correction
    # (count + 1) / (n + 1) — never returns 0 even when no null exceeds
    # the observed, which would mis-state significance under finite-sample
    # permutation testing (Phipson & Smyth 2010). Post-W4 fix 2026-06-12.
    p_value = float((1.0 + np.sum(nulls >= rho_ens_mean)) / (len(nulls) + 1.0))

    # ---- 6. Bootstrap CI over emotions ----
    ci_lo, ci_hi = bootstrap_ci_emotion_block(
        lma_reg, ens_reg, n_boot=args.n_boot, seed=args.seed
    )

    # ---- 7. Report ----
    print()
    print("=" * 70)
    print("Ensemble LMA faithfulness (R1-4)")
    print("=" * 70)
    print(f"  Spearman ρ (ensemble shuffle-mask vs LMA regions): "
          f"{rho_ens_mean:+.4f}")
    print(f"  95% bootstrap CI (emotion block, n=12):           "
          f"[{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  Classical exp079 baseline:                        "
          f"{rho_base_mean:+.4f}")
    print(f"  Δ vs classical baseline:                          "
          f"{rho_ens_mean - rho_base_mean:+.4f}")
    print(f"  Permutation null (n={args.n_perm}): mean = {null_mean:+.4f}, "
          f"std = {null_std:.4f}")
    print(f"  P(null ≥ observed) = {p_value:.4f}")
    print()
    print("Per-emotion ρ:")
    for c, e in enumerate(ORDER):
        print(f"  {e:>10s}: ensemble={rho_ens[c]:+.3f}  "
              f"exp079_baseline={rho_base[c]:+.3f}")

    out = {
        "method": "ensemble_shuffle_mask_to_lma_region",
        "ensemble": "7way_prod_logitmean",
        "mask_mode": "shuffle",
        "spearman_mean": rho_ens_mean,
        "spearman_ci95": [ci_lo, ci_hi],
        "spearman_exp079_baseline": rho_base_mean,
        "spearman_delta_vs_baseline": rho_ens_mean - rho_base_mean,
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
        "per_emotion_spearman_ensemble": {
            e: float(rho_ens[c]) for c, e in enumerate(ORDER)
        },
        "per_emotion_spearman_exp079": {
            e: float(rho_base[c]) for c, e in enumerate(ORDER)
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\n[exp091] saved {args.out_json}")


if __name__ == "__main__":
    main()
