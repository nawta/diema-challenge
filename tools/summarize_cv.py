"""Summarize CV results to markdown table.

Usage:
    python tools/summarize_cv.py
    python tools/summarize_cv.py --experiment exp001_benchmark_repro

See: diema_challenge_implementation_spec.md §12, Makefile summarize
"""

import json
import click
from pathlib import Path

from utils.env import EnvConfig


@click.command()
@click.option("--experiment", type=str, default=None, help="Specific experiment to summarize")
def main(experiment: str):
    """Summarize CV results across folds."""
    env = EnvConfig()

    if experiment:
        experiments = [experiment]
    else:
        experiments = sorted([
            d.name for d in env.artifacts_dir.iterdir()
            if d.is_dir() and d.name.startswith("exp")
        ])

    if not experiments:
        click.echo("No experiment results found.")
        return

    click.echo("| Experiment | Folds | Mean Acc | Std Acc | Mean F1 | Std F1 |")
    click.echo("|------------|-------|---------|---------|---------|--------|")

    for exp_name in experiments:
        exp_dir = env.artifacts_dir / exp_name
        fold_dirs = sorted(exp_dir.glob("fold_*"))

        val_accs = []
        test_accs = []
        test_f1s = []
        for fold_dir in fold_dirs:
            metrics_path = fold_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                if "best_val_acc" in metrics:
                    val_accs.append(metrics["best_val_acc"])
                test_res = metrics.get("test_results", {})
                if "test_acc" in test_res:
                    test_accs.append(test_res["test_acc"])
                if "test_f1" in test_res:
                    test_f1s.append(test_res["test_f1"])

        if test_accs:
            import numpy as np
            mean_acc = np.mean(test_accs) * 100
            std_acc = np.std(test_accs) * 100
            mean_f1 = np.mean(test_f1s) * 100 if test_f1s else 0
            std_f1 = np.std(test_f1s) * 100 if test_f1s else 0
            click.echo(
                f"| {exp_name} | {len(fold_dirs):5d} | "
                f"{mean_acc:5.2f}%  | {std_acc:5.2f}%  | "
                f"{mean_f1:5.2f}%  | {std_f1:5.2f}% |"
            )

            # Per-fold details
            click.echo(f"\n  Per-fold details for {exp_name}:")
            for i, fold_dir in enumerate(fold_dirs):
                mp = fold_dir / "metrics.json"
                if mp.exists():
                    with open(mp) as f:
                        m = json.load(f)
                    tr = m.get("test_results", {})
                    click.echo(
                        f"    {fold_dir.name}: val_acc={m.get('best_val_acc', 0):.4f}, "
                        f"test_acc={tr.get('test_acc', 0):.4f}, "
                        f"test_f1={tr.get('test_f1', 0):.4f}"
                    )
            click.echo()
        else:
            click.echo(f"| {exp_name} | {len(fold_dirs):5d} | -       | -       | -       | -      |")


if __name__ == "__main__":
    main()
