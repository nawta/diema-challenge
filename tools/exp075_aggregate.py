"""Aggregate exp075 head-chain-only ablation results.

Reads metrics.json + oof_logits.npy / oof_labels.npy from
``output/artifacts/exp075_headchain_only_ablation_{variant}/fold_{NN}/``
and prints a comparison table (per-fold F1 + 3-fold mean ± std) and a
plan §5.3 Go-criteria check.

Variants:
    a00_top4         keep_joints = [5,6,7,8]
    a01_top10        keep_joints = top-4 + [10,14,17,18,20,22]
    a02_top10_arms   keep_joints = top-10 + [11,12,15,16]
    a03_top10_legs   keep_joints = top-10 + [19,23,24]

Go criteria (plan §5.3 / §11.5):
    (A) all variants stable: per-fold F1 > 0.10 (no chance-level collapse)
    (B) top-4 substantially ≫ 8.33% chance: F1 ≥ 0.20
    (C) monotone in joint count: top-4 < top-10 ≤ top-10+{arms|legs}

See: §5.3 / §11.5
Related: tools/exp069_ensemble_matrix_8way.py (matrix tool template),
         tools/exp074_aggregate.py (peer aggregator for option B).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ARTIFACT_ROOT = Path("output/artifacts")
EXP_PREFIX = "exp075_headchain_only_ablation"
VARIANTS = [
    ("a00_top4", "top-4 (head-chain)", 4),
    ("a01_top10", "top-10 (head-chain + ranks #5-#10)", 10),
    ("a02_top10_arms", "top-10 + arms (foreArm/Hand)", 14),
    ("a03_top10_legs", "top-10 + legs (Foot/LeftToeBase)", 13),
]
CHANCE_F1 = 1.0 / 12.0


def load_fold(variant: str, fold: int) -> dict:
    base = ARTIFACT_ROOT / f"{EXP_PREFIX}_{variant}" / f"fold_{fold:02d}"
    metrics_path = base / "metrics.json"
    if not metrics_path.exists():
        return {"variant": variant, "fold": fold, "exists": False}
    with metrics_path.open("r") as f:
        m = json.load(f)
    test_f1 = m.get("test_results", {}).get("test_f1")
    test_acc = m.get("test_results", {}).get("test_acc")

    # Verify against OOF arrays for paranoia
    oof_logits_path = base / "oof_logits.npy"
    oof_labels_path = base / "oof_labels.npy"
    if oof_logits_path.exists() and oof_labels_path.exists():
        logits = np.load(oof_logits_path)
        labels = np.load(oof_labels_path)
        preds = logits.argmax(axis=-1)
        oof_f1 = float(f1_score(labels, preds, average="macro"))
        oof_acc = float((preds == labels).mean())
    else:
        oof_f1 = None
        oof_acc = None
    return {
        "variant": variant,
        "fold": fold,
        "exists": True,
        "test_f1": test_f1,
        "test_acc": test_acc,
        "oof_f1": oof_f1,
        "oof_acc": oof_acc,
    }


def main() -> int:
    rows = {}
    for variant, _label, n_joints in VARIANTS:
        rows[variant] = {"n_joints": n_joints, "folds": {}}
        for fold in (0, 1, 2):
            r = load_fold(variant, fold)
            rows[variant]["folds"][fold] = r

    # Print per-fold table
    print("=" * 92)
    print(f"{'variant':<22}{'n_joints':>10}{'fold0':>10}{'fold1':>10}{'fold2':>10}"
          f"{'mean':>10}{'std':>10}")
    print("-" * 92)
    summaries = {}
    for variant, label, n_joints in VARIANTS:
        f1s = []
        cells = []
        for fold in (0, 1, 2):
            r = rows[variant]["folds"][fold]
            if r["exists"] and r["test_f1"] is not None:
                f1s.append(r["test_f1"])
                cells.append(f"{r['test_f1']*100:>9.3f}")
            else:
                cells.append("       --")
        if len(f1s) == 3:
            mean = float(np.mean(f1s))
            std = float(np.std(f1s, ddof=1))
            summary = f"{mean*100:>9.3f}{std*100:>10.3f}"
        else:
            mean = std = None
            summary = "       --        --"
        summaries[variant] = {
            "n_joints": n_joints,
            "label": label,
            "f1s": f1s,
            "mean": mean,
            "std": std,
        }
        print(f"{variant:<22}{n_joints:>10}{cells[0]}{cells[1]}{cells[2]}{summary}")
    print("=" * 92)
    print(f"chance F1 (uniform 12-class): {CHANCE_F1*100:.3f}%")
    print()

    # Plan §5.3 / §11.5 Go-criteria check
    print("Go-criteria check (plan §5.3 / §11.5):")
    print("-" * 92)

    a_ok = True
    a_failures = []
    for variant, _label, _n in VARIANTS:
        for fold in (0, 1, 2):
            r = rows[variant]["folds"][fold]
            if not r["exists"] or r["test_f1"] is None:
                continue
            if r["test_f1"] < 0.10:
                a_ok = False
                a_failures.append(
                    f"  {variant} fold{fold}: F1={r['test_f1']*100:.2f}% (chance {CHANCE_F1*100:.2f}%)"
                )
    print(f"(A) Stability: all per-fold F1 > 10% → {'PASS' if a_ok else 'FAIL'}")
    if not a_ok:
        for line in a_failures:
            print(line)

    # Plan-registered threshold for criterion B is 25% (next_todo_plan §11.5 line
    # 658 / §5.3 line 900: "top-4 が 25% F1 以上"). The smoke prints both 20%
    # and 25% checks so reviewers can see how close we are to the
    # higher pre-registered bar.
    B_THRESHOLD_PLAN = 0.25
    B_THRESHOLD_DRAFT = 0.20  # earlier informal threshold; logged for context only

    top4 = summaries["a00_top4"]

    def _pct(x):
        return "n/a" if x is None else f"{x*100:.3f}%"

    if top4["mean"] is None:
        b_ok = False
        b_str = "  top-4 mean unavailable"
    else:
        b_ok = top4["mean"] >= B_THRESHOLD_PLAN
        b_str = (
            f"  top-4 mean F1 = {_pct(top4['mean'])} "
            f"(plan-registered threshold ≥ {B_THRESHOLD_PLAN*100:.0f}%, "
            f"chance {CHANCE_F1*100:.2f}%); "
            f"draft threshold ≥ {B_THRESHOLD_DRAFT*100:.0f}% "
            f"{'PASSES' if top4['mean'] >= B_THRESHOLD_DRAFT else 'fails'} too"
        )
    print(f"(B) Top-4 ≫ chance: {'PASS' if b_ok else 'FAIL'}")
    print(b_str)

    top10 = summaries["a01_top10"]
    arms = summaries["a02_top10_arms"]
    legs = summaries["a03_top10_legs"]

    # Plan §5.3 / §11.5 specifies "F1 monotone in joint count" without a
    # pre-registered tolerance. Strict monotonicity is the literal check;
    # a ±0.5pp informal tolerance was used in the draft verdict but is NOT
    # plan-registered, so we report both.
    strict_c_ok = (
        top4["mean"] is not None
        and top10["mean"] is not None
        and arms["mean"] is not None
        and legs["mean"] is not None
        and top4["mean"] < top10["mean"]
        and top10["mean"] <= arms["mean"]
        and top10["mean"] <= legs["mean"]
    )
    tolerant_c_ok = (
        top4["mean"] is not None
        and top10["mean"] is not None
        and top4["mean"] < top10["mean"]
        and (
            (arms["mean"] is None or top10["mean"] <= arms["mean"] + 0.005)
            and (legs["mean"] is None or top10["mean"] <= legs["mean"] + 0.005)
        )
    )
    c_str = (
        f"  top4={_pct(top4['mean'])}, top10={_pct(top10['mean'])}, "
        f"arms={_pct(arms['mean'])}, legs={_pct(legs['mean'])}"
    )
    if strict_c_ok:
        c_label = "PASS (strict)"
    elif tolerant_c_ok:
        c_label = "PARTIAL (within ±0.5pp informal tolerance; plan does not pre-register tolerance)"
    else:
        c_label = "FAIL"
    print(f"(C) Monotone in joint count (top4 < top10 < top10+arms / top10+legs): {c_label}")
    print(c_str)
    print()
    # For overall_go we use the strict plan-registered reading (no tolerance).
    c_ok = strict_c_ok

    overall_go = a_ok and b_ok and c_ok
    print(f"Overall: {'GO (paper artifact ready)' if overall_go else 'NO-GO / NEEDS_INSPECTION'}")
    print()

    # Compare top-4 to chance with z-test approximation (per-fold)
    if top4["mean"] is not None:
        delta = top4["mean"] - CHANCE_F1
        print(
            f"top-4 vs chance: Δ = +{delta*100:.3f}pp "
            f"(top-4 mean = {top4['mean']*100:.3f}%, chance = {CHANCE_F1*100:.3f}%)"
        )

    # Save JSON for downstream tooling
    out_path = Path("docs/analysis/exp075_aggregate.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "summaries": summaries,
            "rows": {
                v: {"folds": {str(k): v2 for k, v2 in d["folds"].items()},
                    "n_joints": d["n_joints"]}
                for v, d in rows.items()
            },
            "criteria": {
                "A_stability": a_ok,
                "B_top4_above_chance": b_ok,
                "C_monotone": c_ok,
                "C_monotone_strict": strict_c_ok,
                "C_monotone_tolerant_0p5pp": tolerant_c_ok,
                "overall_go": overall_go,
                "B_threshold_plan": B_THRESHOLD_PLAN,
                "B_threshold_draft": B_THRESHOLD_DRAFT,
            },
            "chance_f1": CHANCE_F1,
        }, f, indent=2)
    print(f"Saved JSON to {out_path}")
    return 0 if overall_go else 1


if __name__ == "__main__":
    sys.exit(main())
