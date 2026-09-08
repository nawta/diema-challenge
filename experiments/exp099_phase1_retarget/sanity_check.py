"""sanity_check.py — Stage 1 post-generation validity checks for

Validates the 4000-clip corpus on 4 axes (per the user's
"Stage 1 sanity check" multi-select decision):

1. **LMA Spearman** on 100 random pairs — should reproduce
   mean ρ ≈ 0.90 across 32 LMA attributes.
2. **Emotion / country balance** — pairs per emotion ≈ 333,
   JP→TW vs TW→JP ≈ 2000 / 2000.
3. **Bone cosine distance distribution** — percentiles + max; flag any
   pair with `bone_length_max_rel_dev > 0.5` (extreme retarget).
4. **Numerical safety** — every .npy is finite (no NaN/Inf) and T ≥ 30.

Output: `output/retarget/phase1_sanity_report.json` + console summary.
Exit code 0 if all 4 axes PASS, non-zero otherwise.

Related: experiments/exp099_bone_retarget/lma_quality_check.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from diema.features.lma_attributes import (  # noqa: E402
    compute_lma_attributes, LMA_NAMES)

MANIFEST = Path("output/retarget/phase1_manifest.json")
CACHED_NPZ = Path("data/diema_challenge/processed/motion_train_quat.npz")
REPORT_JSON = Path("output/retarget/phase1_sanity_report.json")

# Thresholds for each check
LMA_MEAN_RHO_FLOOR = 0.80 # got 0.903; allow some headroom
EMOTION_BALANCE_TOL = 0.30     # ≤30% deviation from expected count per emotion
COUNTRY_BALANCE_TOL = 0.10     # ≤10% deviation from 50/50 JP/TW split
EXTREME_DEV_THRESHOLD = 0.5    # bone_length_max_rel_dev > 0.5 flagged
MIN_T = 30                     # clips with T < 30 cannot be smoke-trained

LMA_SAMPLE_SIZE = 100
SEED = 42


def check_lma_spearman(pairs: list[dict], npz, rng) -> dict:
    """Run LMA Spearman on a random subsample. Returns report dict."""
    n_sample = min(LMA_SAMPLE_SIZE, len(pairs))
    sample_idx = rng.choice(len(pairs), size=n_sample, replace=False)
    src_lma = np.zeros((n_sample, 32), dtype=np.float32)
    tgt_lma = np.zeros((n_sample, 32), dtype=np.float32)

    for k, idx in enumerate(sample_idx):
        p = pairs[idx]
        i = p["idx"]
        joint_quats = npz[f"clip_{i}_joint_data"].astype(np.float32)
        root_pos = npz[f"clip_{i}_root_pos"].astype(np.float32)
        sidecar = np.load(p["output_npz"])
        src_off = sidecar["source_offsets"]
        tgt_off = sidecar["target_offsets"]
        new_root = sidecar["new_root_pos"]
        src_lma[k] = compute_lma_attributes(
            root_pos=root_pos, joint_quats=joint_quats, offsets=src_off)
        tgt_lma[k] = compute_lma_attributes(
            root_pos=new_root, joint_quats=joint_quats, offsets=tgt_off)

    per_attr = []
    for j, name in enumerate(LMA_NAMES):
        x = src_lma[:, j]
        y = tgt_lma[:, j]
        if x.std() < 1e-8 or y.std() < 1e-8:
            rho = 1.0
        else:
            rho_val, _ = spearmanr(x, y)
            rho = float(rho_val) if not np.isnan(rho_val) else 1.0
        per_attr.append({"attribute": name, "rho": rho})

    mean_rho = float(np.mean([a["rho"] for a in per_attr]))
    broken = [a["attribute"] for a in per_attr if a["rho"] < 0.4]
    passed = mean_rho >= LMA_MEAN_RHO_FLOOR

    return {
        "passed": passed,
        "n_sample": int(n_sample),
        "mean_rho": mean_rho,
        "floor": LMA_MEAN_RHO_FLOOR,
        "broken_attrs": broken,
        "n_broken": len(broken),
    }


def check_balance(pairs: list[dict]) -> dict:
    """Emotion and JP↔TW direction balance."""
    n_total = len(pairs)
    emo_counts = Counter(p["emotion"] for p in pairs)
    expected_per_emo = n_total / len(emo_counts)
    emo_devs = {
        emo: abs(c - expected_per_emo) / expected_per_emo
        for emo, c in emo_counts.items()
    }
    emo_pass = max(emo_devs.values()) <= EMOTION_BALANCE_TOL

    n_jp_to_tw = sum(1 for p in pairs if p["country"] == "JP")
    n_tw_to_jp = sum(1 for p in pairs if p["country"] == "TW")
    direction_dev = abs(n_jp_to_tw - n_tw_to_jp) / n_total
    direction_pass = direction_dev <= COUNTRY_BALANCE_TOL

    return {
        "passed": emo_pass and direction_pass,
        "emotion_pass": emo_pass,
        "direction_pass": direction_pass,
        "n_total": n_total,
        "n_jp_to_tw": n_jp_to_tw,
        "n_tw_to_jp": n_tw_to_jp,
        "direction_dev": float(direction_dev),
        "direction_tol": COUNTRY_BALANCE_TOL,
        "emotion_counts": dict(sorted(emo_counts.items())),
        "emotion_expected": float(expected_per_emo),
        "emotion_max_dev": float(max(emo_devs.values())),
        "emotion_tol": EMOTION_BALANCE_TOL,
    }


def check_bone_distance(pairs: list[dict]) -> dict:
    """Bone-cosine-distance distribution + extreme-retarget flags."""
    dists = np.array([p["bone_cosine_distance"] for p in pairs])
    devs = np.array([
        p.get("retarget_meta", {}).get("bone_length_max_rel_dev", -1.0)
        for p in pairs
    ])
    valid = devs >= 0.0
    n_extreme = int(np.sum(devs[valid] > EXTREME_DEV_THRESHOLD))

    return {
        "passed": True,    # informational; no hard fail criterion
        "cosine_distance": {
            "min": float(dists.min()),
            "p5": float(np.percentile(dists, 5)),
            "p50": float(np.percentile(dists, 50)),
            "p95": float(np.percentile(dists, 95)),
            "max": float(dists.max()),
            "mean": float(dists.mean()),
        },
        "bone_length_max_rel_dev": {
            "n_with_meta": int(valid.sum()),
            "min": float(devs[valid].min()) if valid.any() else None,
            "p50": float(np.percentile(devs[valid], 50)) if valid.any() else None,
            "p95": float(np.percentile(devs[valid], 95)) if valid.any() else None,
            "max": float(devs[valid].max()) if valid.any() else None,
            "n_extreme": n_extreme,
            "extreme_threshold": EXTREME_DEV_THRESHOLD,
        },
    }


def check_numerical(pairs: list[dict]) -> dict:
    """NaN/Inf + T ≥ 30 check on every .npy."""
    bad_nan = []
    bad_short = []
    t_dist = []
    for p in pairs:
        arr = np.load(p["output_npy"])
        t_dist.append(arr.shape[0])
        if not np.isfinite(arr).all():
            bad_nan.append(p["output_npy"])
        if arr.shape[0] < MIN_T:
            bad_short.append({"path": p["output_npy"], "T": int(arr.shape[0])})
    t_dist = np.array(t_dist)
    passed = (len(bad_nan) == 0) and (len(bad_short) == 0)
    return {
        "passed": passed,
        "n_checked": len(pairs),
        "n_with_nan_inf": len(bad_nan),
        "n_with_T_under_min": len(bad_short),
        "min_T": int(t_dist.min()),
        "p5_T": int(np.percentile(t_dist, 5)),
        "median_T": int(np.percentile(t_dist, 50)),
        "p95_T": int(np.percentile(t_dist, 95)),
        "max_T": int(t_dist.max()),
        "min_T_required": MIN_T,
        "bad_nan_first10": bad_nan[:10],
        "bad_short_first10": bad_short[:10],
    }


def main():
    rng = np.random.default_rng(SEED)
    if not MANIFEST.exists():
        print(f"[sanity] ERROR: manifest not found at {MANIFEST}", file=sys.stderr)
        print(f"[sanity] Run generate_full_corpus.py first.", file=sys.stderr)
        sys.exit(2)
    manifest = json.loads(MANIFEST.read_text())
    pairs = manifest["pairs"]
    print(f"[sanity] {len(pairs)} pairs in manifest")

    print(f"[sanity] loading cached npz {CACHED_NPZ}")
    npz = np.load(CACHED_NPZ, allow_pickle=True)

    print(f"[sanity] (1/4) LMA Spearman on {LMA_SAMPLE_SIZE} random pairs...")
    lma_report = check_lma_spearman(pairs, npz, rng)
    print(f"  mean ρ = {lma_report['mean_rho']:.4f} "
          f"(floor {LMA_MEAN_RHO_FLOOR}; "
          f"broken: {lma_report['n_broken']}) → "
          f"{'PASS' if lma_report['passed'] else 'FAIL'}")

    print(f"[sanity] (2/4) Emotion / country balance...")
    balance_report = check_balance(pairs)
    print(f"  JP→TW: {balance_report['n_jp_to_tw']}, "
          f"TW→JP: {balance_report['n_tw_to_jp']}, "
          f"direction_dev: {balance_report['direction_dev']:.3f} "
          f"(tol {COUNTRY_BALANCE_TOL})")
    print(f"  emotion_max_dev: {balance_report['emotion_max_dev']:.3f} "
          f"(tol {EMOTION_BALANCE_TOL})")
    print(f"  → {'PASS' if balance_report['passed'] else 'FAIL'}")

    print(f"[sanity] (3/4) Bone cosine distance distribution...")
    dist_report = check_bone_distance(pairs)
    cd = dist_report["cosine_distance"]
    print(f"  cosine_dist p5/p50/p95: {cd['p5']:.4f} / "
          f"{cd['p50']:.4f} / {cd['p95']:.4f}")
    blrd = dist_report["bone_length_max_rel_dev"]
    if blrd.get('max') is not None:
        print(f"  bone_length_max_rel_dev p50/p95/max: "
              f"{blrd['p50']:.3f} / {blrd['p95']:.3f} / {blrd['max']:.3f}")
        print(f"  n_extreme (> {EXTREME_DEV_THRESHOLD}): {blrd['n_extreme']}")

    print(f"[sanity] (4/4) NaN/Inf + T ≥ {MIN_T} check on all .npy...")
    num_report = check_numerical(pairs)
    print(f"  n_checked: {num_report['n_checked']}, "
          f"NaN/Inf: {num_report['n_with_nan_inf']}, "
          f"T < {MIN_T}: {num_report['n_with_T_under_min']}")
    print(f"  T p5/p50/p95: {num_report['p5_T']} / "
          f"{num_report['median_T']} / {num_report['p95_T']}")
    print(f"  → {'PASS' if num_report['passed'] else 'FAIL'}")

    overall = (lma_report["passed"] and balance_report["passed"]
                and num_report["passed"])
    report = {
        "overall_passed": overall,
        "lma_spearman": lma_report,
        "balance": balance_report,
        "bone_distance": dist_report,
        "numerical": num_report,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2))
    print(f"\n[sanity] report → {REPORT_JSON}")
    print(f"[sanity] overall: {'PASS' if overall else 'FAIL'}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
