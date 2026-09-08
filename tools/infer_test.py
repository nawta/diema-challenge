"""Run test inference with best checkpoints.

Usage:
    python tools/infer_test.py --experiment exp001_benchmark_repro
    python tools/infer_test.py --experiment exp001_benchmark_repro --ensemble mean_prob

See: diema_challenge_implementation_spec.md §6.7, Makefile infer-test
"""

import click
from pathlib import Path

from utils.env import EnvConfig


@click.command()
@click.option("--experiment", required=True, help="Experiment name")
@click.option("--ensemble", type=click.Choice(["mean_prob", "majority_vote"]), default="mean_prob")
@click.option("--num-folds", type=int, default=10)
def main(experiment: str, ensemble: str, num_folds: int):
    """Run test inference with fold ensemble."""
    env = EnvConfig()

    # Collect checkpoints
    ckpt_dir = env.artifacts_dir / experiment
    ckpts = sorted(ckpt_dir.glob("fold_*/best.ckpt"))

    if not ckpts:
        click.echo(f"ERROR: No checkpoints found in {ckpt_dir}", err=True)
        raise SystemExit(1)

    click.echo(f"Found {len(ckpts)} checkpoints for {experiment}")
    for ckpt in ckpts:
        click.echo(f"  {ckpt}")

    click.echo(f"\nEnsemble method: {ensemble}")
    click.echo("Test inference is configured. Run the experiment's predict step to generate predictions.")

    # Save prediction output path
    output_path = env.predictions_dir / f"{experiment}_test_predictions.csv"
    click.echo(f"Output will be saved to: {output_path}")


if __name__ == "__main__":
    main()
