"""Run K-fold cross-validation for an experiment.

Usage:
    python tools/run_cv.py --experiment exp001_benchmark_repro --num-folds 10
    python tools/run_cv.py --experiment exp001_benchmark_repro --folds 0 1 2

See: diema_challenge_implementation_spec.md §6.5, Makefile cv-baseline
"""

import click
import subprocess
import sys


@click.command()
@click.option("--experiment", required=True, help="Experiment folder name (e.g., exp001_benchmark_repro)")
@click.option("--num-folds", type=int, default=10, help="Total number of folds")
@click.option("--folds", type=int, multiple=True, default=None, help="Specific folds to run (default: all)")
@click.option("--exp-version", type=str, default="000", help="Minor version (exp/*.yaml)")
def main(experiment: str, num_folds: int, folds: tuple, exp_version: str):
    """Run K-fold CV by invoking experiment run.py for each fold."""
    if not folds:
        folds = tuple(range(num_folds))

    click.echo(f"Running {len(folds)}-fold CV for {experiment} (version {exp_version})")

    for fold_idx in folds:
        click.echo(f"\n{'='*60}")
        click.echo(f"  Fold {fold_idx}/{num_folds - 1}")
        click.echo(f"{'='*60}")

        cmd = [
            sys.executable, "-m", f"experiments.{experiment}.run",
            f"fold={fold_idx}",
            f"num_folds={num_folds}",
        ]

        result = subprocess.run(cmd, cwd=".")
        if result.returncode != 0:
            click.echo(f"ERROR: Fold {fold_idx} failed with return code {result.returncode}", err=True)
            raise SystemExit(result.returncode)

    click.echo(f"\nAll {len(folds)} folds completed.")


if __name__ == "__main__":
    main()
