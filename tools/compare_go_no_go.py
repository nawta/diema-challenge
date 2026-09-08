""" Go/No-Go comparison between a new experiment and a baseline.

See: (and any other Phase with a 3-fold Go gate).

Loads ``metrics.json`` from each experiment's ``fold_XX/`` directories,
aligns the two experiments on the fold ids they share, and prints a table
of per-fold val-F1 + the mean delta. Exits 0 if the delta meets the
``--threshold`` (default +0.3pt = 0.003 absolute); exits 1 otherwise so a
CI / batch driver can gate the 10-fold run on this.

Usage::

    python tools/compare_go_no_go.py \\
        --baseline exp020_conv1d_transformer_a01 \\
        --candidate exp051_tmr_scenario_global_a00 \\
        --threshold 0.003
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.env import EnvConfig  # noqa: E402


def _collect_fold_metrics(exp_dir: Path) -> dict[int, dict]:
    """Return ``{fold_id: metrics_dict}`` for all fold_XX with metrics.json."""
    out: dict[int, dict] = {}
    for p in sorted(exp_dir.glob("fold_[0-9][0-9]")):
        m = p / "metrics.json"
        if not m.exists():
            continue
        try:
            data = json.loads(m.read_text())
        except json.JSONDecodeError:
            continue
        fold_id = int(p.name.split("_")[1])
        out[fold_id] = data
    return out


def _best_f1(metrics: dict) -> float | None:
    """Extract the val-F1 the experiment calls "best" from metrics.json.

    We prefer ``test_results.test_f1`` since the training loop calls
    ``trainer.test()`` on the val-fold with ``ckpt_path="best"`` right after
    ``fit()``, so that value reflects the F1 on the best-val-F1 checkpoint.
    Falls back to the max over the history if test_results is missing.
    """
    tr = metrics.get("test_results")
    if isinstance(tr, dict) and "test_f1" in tr:
        return float(tr["test_f1"])
    history = metrics.get("history", [])
    vals = [h.get("val_f1") for h in history if "val_f1" in h]
    return max(vals) if vals else None


@click.command()
@click.option("--baseline", required=True, help="Baseline experiment folder name")
@click.option("--candidate", required=True, help="Candidate experiment folder name")
@click.option("--threshold", default=0.003, type=float,
              help="Candidate-minus-baseline mean F1 threshold in absolute units (default 0.003 = 0.3pt)")
@click.option("--max-folds", default=None, type=int,
              help="Limit to first N matching folds (Go/No-Go usually 3)")
def main(baseline: str, candidate: str, threshold: float,
         max_folds: int | None) -> None:
    env = EnvConfig()
    base_dir = env.artifacts_dir / baseline
    cand_dir = env.artifacts_dir / candidate
    if not base_dir.exists():
        click.echo(f"ERROR: baseline dir not found: {base_dir}", err=True)
        raise SystemExit(2)
    if not cand_dir.exists():
        click.echo(f"ERROR: candidate dir not found: {cand_dir}", err=True)
        raise SystemExit(2)

    base_m = _collect_fold_metrics(base_dir)
    cand_m = _collect_fold_metrics(cand_dir)
    common = sorted(set(base_m) & set(cand_m))
    if max_folds:
        common = common[:max_folds]
    if not common:
        click.echo("ERROR: no overlapping fold metrics.json in both experiments", err=True)
        raise SystemExit(2)

    click.echo(f"\nBaseline : {baseline}")
    click.echo(f"Candidate: {candidate}")
    click.echo(f"Folds    : {common}\n")
    click.echo(f"{'fold':<6} {'baseline_f1':>12} {'candidate_f1':>12} {'Δ (pt)':>10}")
    click.echo("-" * 44)
    deltas: list[float] = []
    base_vals: list[float] = []
    cand_vals: list[float] = []
    for f in common:
        b = _best_f1(base_m[f])
        c = _best_f1(cand_m[f])
        if b is None or c is None:
            click.echo(f"{f:<6} {'(missing)':>12}")
            continue
        d = c - b
        deltas.append(d)
        base_vals.append(b)
        cand_vals.append(c)
        click.echo(f"{f:<6} {b:>12.4f} {c:>12.4f} {d*100:>+9.2f}")
    click.echo("-" * 44)
    b_mean = sum(base_vals) / len(base_vals)
    c_mean = sum(cand_vals) / len(cand_vals)
    d_mean = c_mean - b_mean
    click.echo(f"{'mean':<6} {b_mean:>12.4f} {c_mean:>12.4f} {d_mean*100:>+9.2f}")

    verdict = "GO" if d_mean >= threshold else "NO-GO"
    click.echo(f"\nThreshold: +{threshold*100:.2f}pt  ⇒  Verdict: {verdict}")
    raise SystemExit(0 if d_mean >= threshold else 1)


if __name__ == "__main__":
    main()
