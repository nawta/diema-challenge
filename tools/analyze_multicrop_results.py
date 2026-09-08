""" exp068 post-sweep analysis: rank settings vs single-crop baseline.

Reads ``experiments/exp068_multicrop_inference/results.json`` (produced by
``tools/infer_test_multicrop.py``) and produces a ranked summary table with
Δ vs the existing single-crop 7-way baseline OOF F1, plus per-fold std
ratio and Go-criterion checks (criterion 1: ΔF1 ≥ +0.30pp;
criterion 2: per-fold std ratio ≤ 1.5).

For criterion 3 (sample-level paired bootstrap CI), use the separate
``tools/multicrop_bootstrap.py`` driver — that script re-runs inference
for ONE specific (policy, num_crops, aggregation) and computes the paired
bootstrap directly. This script does not perform that bootstrap because
the sweep ``results.json`` only stores summary metrics, not per-sample
aggregated OOF logits.

CLI usage::

    python tools/analyze_multicrop_results.py \
        --results experiments/exp068_multicrop_inference/results.json

Output (under ``experiments/exp068_multicrop_inference/analysis/``):
  * settings_ranking.json — ranked table with Δ, std ratio, criterion checks
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.env import EnvConfig  # noqa: E402
from diema.data.parser import IDX_TO_EMOTION  # noqa: E402

from tools.calibrate_ensemble import (  # noqa: E402
    DEFAULT_ENSEMBLE,
    load_ensemble_oof,
    macro_f1,
    paired_bootstrap_f1_diff,
    per_class_f1,
)


@click.command()
@click.option(
    "--results",
    default="experiments/exp068_multicrop_inference/results.json",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--out-dir",
    default="experiments/exp068_multicrop_inference/analysis",
    type=click.Path(file_okay=False),
)
def main(results: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(results) as fh:
        data = json.load(fh)

    settings = data["settings"]
    if not settings:
        raise click.ClickException("results.json has no settings; sweep failed?")

    # Compute single-crop baseline F1 from saved OOF logits.
    env = EnvConfig()
    folds = load_ensemble_oof(env.artifacts_dir, DEFAULT_ENSEMBLE, num_folds=10)
    base_logits = np.concatenate([f.logits for f in folds], axis=0)
    base_labels = np.concatenate([f.labels for f in folds], axis=0)
    base_preds = base_logits.argmax(axis=1)
    base_f1 = macro_f1(base_preds, base_labels)
    base_per_fold_f1 = [
        macro_f1(f.logits.argmax(axis=1), f.labels) for f in folds
    ]
    base_per_fold_std = float(np.std(base_per_fold_f1, ddof=1))
    base_per_class = per_class_f1(base_preds, base_labels)

    click.echo(
        f"Single-crop 7-way baseline: F1 = {base_f1*100:.3f}%, "
        f"per-fold std = {base_per_fold_std*100:.3f}pp"
    )

    # Rank settings; compute Δ vs baseline.
    ranked = []
    for s in settings:
        delta = s["overall_f1"] - base_f1
        # Distribution-shift signal: per-fold std ratio vs baseline.
        std_ratio = (
            (s["per_fold_f1_std"] / base_per_fold_std)
            if base_per_fold_std > 0 else float("inf")
        )
        ranked.append({
            **s,
            "delta_f1_pp": float(delta * 100.0),
            "per_fold_f1_std_pp": float(s["per_fold_f1_std"] * 100.0),
            "per_fold_std_ratio": float(std_ratio),
            "criterion_1_plus_0.30pt": bool(delta * 100.0 >= 0.30),
            "criterion_2_std_within_50pct": bool(std_ratio <= 1.50),
        })
    ranked.sort(key=lambda r: r["overall_f1"], reverse=True)

    # ----- Header -----
    click.echo(
        f"\n=== {len(ranked)} settings vs single-crop baseline (F1 = {base_f1*100:.3f}%) ==="
    )
    header = (
        f"{'rank':>4}  {'policy':<24}  {'K':>3}  {'agg':<11}  "
        f"{'F1%':>7}  {'Δpp':>7}  {'std%':>6}  {'std_r':>6}  "
        f"{'c1':>3}  {'c2':>3}"
    )
    click.echo(header)
    for i, r in enumerate(ranked, start=1):
        c1 = "✓" if r["criterion_1_plus_0.30pt"] else "✗"
        c2 = "✓" if r["criterion_2_std_within_50pct"] else "✗"
        click.echo(
            f"{i:>4}  {r['policy']:<24}  {r['num_crops']:>3}  {r['aggregation']:<11}  "
            f"{r['overall_f1']*100:>6.3f}  {r['delta_f1_pp']:>+7.3f}  "
            f"{r['per_fold_f1_std_pp']:>5.3f}  {r['per_fold_std_ratio']:>6.3f}  "
            f"{c1:>3}  {c2:>3}"
        )

    # ----- Per-policy summary -----
    click.echo("\n=== Per-policy best (across K, agg) ===")
    by_policy: dict[str, list] = {}
    for r in ranked:
        by_policy.setdefault(r["policy"], []).append(r)
    for policy, rows in by_policy.items():
        rows = sorted(rows, key=lambda x: x["overall_f1"], reverse=True)
        best = rows[0]
        click.echo(
            f"  {policy:<24}: best F1 = {best['overall_f1']*100:.3f}% "
            f"(Δ {best['delta_f1_pp']:+.3f}pp), "
            f"setting: K={best['num_crops']}, agg={best['aggregation']}"
        )

    # ----- Best-overall verdict -----
    best = ranked[0]
    click.echo(
        f"\nBest multi-crop setting: {best['policy']} K={best['num_crops']} "
        f"{best['aggregation']} → F1 {best['overall_f1']*100:.3f}% "
        f"(Δ vs baseline = {best['delta_f1_pp']:+.3f}pp)"
    )

    # Persist
    with open(out / "settings_ranking.json", "w") as fh:
        json.dump({
            "baseline_f1": float(base_f1),
            "baseline_per_fold_f1": list(map(float, base_per_fold_f1)),
            "baseline_per_fold_std": base_per_fold_std,
            "baseline_per_class_f1": {
                IDX_TO_EMOTION[c]: float(base_per_class[c])
                for c in range(12)
            },
            "ranked": ranked,
        }, fh, indent=2, default=_json_default)
    click.echo(f"\nWrote {out / 'settings_ranking.json'}")


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
