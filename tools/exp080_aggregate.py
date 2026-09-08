"""exp080 — aggregate scale_a00 10-fold results and compare to production a00.

Reads each fold's metrics.json + (best test_f1 / test_acc), computes mean ± std,
prints comparison table vs production exp034 a00 (val_f1 30.06% ± 3.29%).

Usage: conda run -n acii2026 python tools/exp080_aggregate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("output/artifacts")
VARIANTS = {
    "production_a00": ROOT / "exp034_regionaware_convtr_a00",
    "scale_a00":       ROOT / "exp034_regionaware_convtr_scale_a00",
    "scale_a01":       ROOT / "exp034_regionaware_convtr_scale_a01",
    "scale_a02":       ROOT / "exp034_regionaware_convtr_scale_a02",
}
PRODUCTION_VAL_F1_MEAN = 30.06   # 10-fold mean from docs/results_summary.md
PRODUCTION_VAL_F1_STD  = 3.29


def collect(folder: Path) -> tuple[list[float], list[float]]:
    """Return per-fold (val_acc, test_f1) lists, sorted by fold idx."""
    val_accs = []
    test_f1s = []
    for fold_dir in sorted(folder.glob("fold_*")):
        if not fold_dir.is_dir() or "preBNfix" in fold_dir.name:
            continue
        metrics = fold_dir / "metrics.json"
        if not metrics.exists():
            continue
        with open(metrics) as f:
            d = json.load(f)
        if "test_results" not in d or "test_f1" not in d["test_results"]:
            continue
        val_accs.append(d["best_val_acc"] * 100)
        test_f1s.append(d["test_results"]["test_f1"] * 100)
    return val_accs, test_f1s


def main() -> None:
    rows = []
    for name, folder in VARIANTS.items():
        v, f = collect(folder)
        if not v:
            rows.append((name, 0, None, None, None, None, []))
            continue
        v_mean, v_std = float(np.mean(v)), float(np.std(v))
        f_mean, f_std = float(np.mean(f)), float(np.std(f))
        rows.append((name, len(v), v_mean, v_std, f_mean, f_std, f))

    print(f"{'variant':20s} {'folds':>6s} {'val_acc mean':>14s} {'val_acc std':>13s} {'test_f1 mean':>14s} {'test_f1 std':>13s}")
    print("-" * 90)
    for name, n, vm, vs, fm, fs, _ in rows:
        if vm is None:
            print(f"{name:20s} {n:>6d} {'--':>14s} {'--':>13s} {'--':>14s} {'--':>13s}")
        else:
            print(f"{name:20s} {n:>6d} {vm:>13.2f}% {vs:>12.2f}% {fm:>13.2f}% {fs:>12.2f}%")

    scale_row = next((r for r in rows if r[0] == "scale_a00"), None)
    prod_row = next((r for r in rows if r[0] == "production_a00"), None)
    if scale_row and scale_row[1] > 0 and prod_row and prod_row[1] > 0:
        scale_f1 = scale_row[6]
        prod_f1 = prod_row[6]
        common_n = min(len(scale_f1), len(prod_f1))
        if common_n > 0:
            diffs = np.array(scale_f1[:common_n]) - np.array(prod_f1[:common_n])
            print()
            print(f"=== scale_a00 vs production_a00 (paired by fold, n={common_n}) ===")
            print(f"  fold-paired mean Δ test_f1: {diffs.mean():+.2f}pp")
            print(f"  fold-paired std Δ:           {diffs.std():.2f}pp")
            print(f"  folds with positive Δ:       {int((diffs > 0).sum())}/{common_n}")
            print(f"  worst fold Δ:                {diffs.min():+.2f}pp")
            print(f"  best fold Δ:                 {diffs.max():+.2f}pp")
            # Crude paired t-test (no scipy assumption)
            mean_diff = diffs.mean()
            sem = diffs.std(ddof=1) / np.sqrt(common_n)
            t_stat = mean_diff / sem if sem > 0 else float("nan")
            print(f"  paired t-stat:               {t_stat:.2f} (df={common_n-1})")
        if scale_row[1] == 10:
            print()
            print(f"=== scale_a00 vs production (docs 10-fold mean {PRODUCTION_VAL_F1_MEAN:.2f}%) ===")
            print(f"  scale_a00 10-fold mean test_f1: {scale_row[4]:.2f}%")
            print(f"  Δ vs production docs mean:      {scale_row[4]-PRODUCTION_VAL_F1_MEAN:+.2f}pp")
            print(f"  GO threshold (≥ baseline + 0.5pp): {'PASS' if scale_row[4] >= PRODUCTION_VAL_F1_MEAN + 0.5 else 'FAIL'}")


if __name__ == "__main__":
    main()
