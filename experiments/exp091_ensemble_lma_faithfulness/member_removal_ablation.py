""" member-removal stratified ablation on the 11-way ensemble.

Related:
- experiments/exp091_ensemble_lma_faithfulness/performer_block_bootstrap.py
    (shared OOF loaders + 11-way fusion math)

Goal: convince the reviewer that the 11-way headline F1 is not driven by
C3D (whose Vicon contact features have been associated with country
leakage in exp056) nor by any single externally-pretrained branch. We
do this with two ablations:

  1. **Leave-one-member-out (LOMO)**: drop one of the 11 members, re-fuse
     the remaining 10 with the same logit-mean convention, recompute
     pooled Macro-F1. Headline: ``Δ vs 11-way`` per dropped member.
  2. **Stratified F1**: 11-way + LOMO Macro-F1 on JP-only, TW-only, and
     OptiTrack-excluded sample subsets. Headline: ``Δ JP / Δ TW``,
     answering "does the ensemble's gain over the baseline survive when
     a country / capture-system stratum is removed?"

We also include the published "minus all external pretrains" pattern —
drop {MotionBERT, MAMP60, MAMP120} simultaneously — to isolate the
contribution of the cross-dataset SSL branches as a group.
"""

from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.env import EnvConfig  # noqa: E402

from experiments.exp091_ensemble_lma_faithfulness.performer_block_bootstrap import (  # noqa: E402
    PROD_MEMBERS, EXTERNAL_OOFS, _softmax,
)
from experiments.exp091_ensemble_lma_faithfulness.logit_scale_ablation import (  # noqa: E402
    _build_11way_member_logits, fuse_logit_mean,
)


NUM_CLASS = 12


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _is_jp(filenames: np.ndarray) -> np.ndarray:
    return np.array([str(fn).startswith("JP_") for fn in filenames])


def _is_tw(filenames: np.ndarray) -> np.ndarray:
    return np.array([str(fn).startswith("TW_") for fn in filenames])


def _is_optitrack(filenames: np.ndarray) -> np.ndarray:
    """Filenames produced via OptiTrack capture.

    Per the OptiTrack subset is the early
    Japanese performers (P01-P12); the rest use the production Vicon
    rig. We approximate this by the performer-id prefix JP_01..JP_12.
    """
    import re
    out = []
    for fn in filenames:
        m = re.match(r"^JP_(\d+)_", str(fn))
        if m and int(m.group(1)) <= 12:
            out.append(True)
        else:
            out.append(False)
    return np.array(out)


def _fuse_subset(member_logits: np.ndarray, keep_idx: list[int]) -> np.ndarray:
    """Re-fuse a subset of members via logit-mean."""
    return fuse_logit_mean(member_logits[keep_idx])


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out-json",
                   default="experiments/exp091_ensemble_lma_faithfulness/r4_results.json")
    args = p.parse_args()

    env = EnvConfig()
    print("[R4] Loading 11-way member OOF logits …")
    t0 = time.time()
    member_logits, labels, fold_ids, filenames, member_names = _build_11way_member_logits(env)
    print(f"[R4] member_logits {member_logits.shape}, "
          f"n_samples {len(labels)} ({time.time()-t0:.1f}s)")

    M = len(member_names)

    # Pre-compute stratum masks.
    masks = {
        "all":              np.ones(len(filenames), dtype=bool),
        "JP_only":          _is_jp(filenames),
        "TW_only":          _is_tw(filenames),
        "OptiTrack_excluded": ~_is_optitrack(filenames),
    }
    for k, m in masks.items():
        print(f"  stratum {k:>20s}: n = {int(m.sum())}")

    results: dict = {}

    # 1. 11-way headline F1 per stratum
    fused_full = _fuse_subset(member_logits, list(range(M)))
    pred_full = fused_full.argmax(1)
    results["headline_11way"] = {
        k: _macro_f1(labels[m], pred_full[m]) for k, m in masks.items()
    }
    print("\n[R4] 11-way Macro-F1 per stratum:")
    for k, v in results["headline_11way"].items():
        print(f"  {k:>20s}: {v*100:.4f} %")

    # 2. Leave-one-member-out F1 + Δ vs 11-way headline (all stratum)
    results["lomo"] = {}
    print("\n[R4] LOMO Δ vs 11-way (Macro-F1, pp):")
    print(f"  {'dropped':>40s}  {'all':>8s}  {'JP':>8s}  {'TW':>8s}  {'-OptiTr':>9s}")
    for i, name in enumerate(member_names):
        keep = [j for j in range(M) if j != i]
        fused = _fuse_subset(member_logits, keep)
        pred = fused.argmax(1)
        f1s = {k: _macro_f1(labels[m], pred[m]) for k, m in masks.items()}
        deltas = {k: (f1s[k] - results["headline_11way"][k]) * 100 for k in masks}
        results["lomo"][name] = {
            "f1_per_stratum": f1s,
            "delta_vs_11way_pp": deltas,
        }
        print(
            f"  −{name:>40s}  "
            f"{deltas['all']:+7.3f}  {deltas['JP_only']:+7.3f}  "
            f"{deltas['TW_only']:+7.3f}  {deltas['OptiTrack_excluded']:+8.3f}"
        )

    # 3. Group ablations
    group_ablations: dict[str, list[str]] = {
        "minus_c3d":               ["c3d"],
        "minus_all_externals":     list(EXTERNAL_OOFS.keys()),
        "minus_motionbert":        ["motionbert"],
        "minus_both_mamps":        ["mamp60", "mamp120"],
        "prod_only_7way":          list(EXTERNAL_OOFS.keys()),  # alias of minus_all_externals
    }
    name_to_idx = {n: i for i, n in enumerate(member_names)}
    results["group_ablations"] = {}
    print("\n[R4] Group ablation Δ vs 11-way (Macro-F1, pp):")
    print(f"  {'pattern':>40s}  {'all':>8s}  {'JP':>8s}  {'TW':>8s}  {'-OptiTr':>9s}")
    for pattern, drop_list in group_ablations.items():
        if not all(n in name_to_idx for n in drop_list):
            continue
        keep = [i for i, n in enumerate(member_names) if n not in drop_list]
        fused = _fuse_subset(member_logits, keep)
        pred = fused.argmax(1)
        f1s = {k: _macro_f1(labels[m], pred[m]) for k, m in masks.items()}
        deltas = {k: (f1s[k] - results["headline_11way"][k]) * 100 for k in masks}
        results["group_ablations"][pattern] = {
            "f1_per_stratum": f1s,
            "delta_vs_11way_pp": deltas,
            "n_members": len(keep),
        }
        print(
            f"  {pattern:>40s}  "
            f"{deltas['all']:+7.3f}  {deltas['JP_only']:+7.3f}  "
            f"{deltas['TW_only']:+7.3f}  {deltas['OptiTrack_excluded']:+8.3f}"
        )

    out = {
        "method": "member_removal_stratified_ablation",
        "ensemble": "11way_logitmean",
        "n_samples_per_stratum": {k: int(m.sum()) for k, m in masks.items()},
        "members": list(member_names),
        "results": results,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\n[R4] saved {args.out_json}")


if __name__ == "__main__":
    main()
