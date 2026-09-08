"""Convert internal predictions to organizer submission format.

Usage:
    python tools/make_submission.py --experiment exp001_benchmark_repro
    python tools/make_submission.py --input output/predictions/xxx.csv --output output/submissions/xxx.csv

See: diema_challenge_implementation_spec.md §6.8, Makefile make-submission
"""

import click
import pandas as pd
from pathlib import Path

from utils.env import EnvConfig
from diema.data.parser import IDX_TO_EMOTION


@click.command()
@click.option("--experiment", type=str, default=None, help="Experiment name (auto-locates prediction file)")
@click.option("--input", "input_path", type=str, default=None, help="Input prediction CSV")
@click.option("--output", "output_path", type=str, default=None, help="Output submission CSV")
@click.option("--include-probabilities", is_flag=True, help="Include probability columns")
def main(experiment: str, input_path: str, output_path: str, include_probabilities: bool):
    """Convert predictions to submission format."""
    env = EnvConfig()

    if input_path is None:
        if experiment is None:
            click.echo("ERROR: Provide --experiment or --input", err=True)
            raise SystemExit(1)
        input_path = str(env.predictions_dir / f"{experiment}_test_predictions.csv")

    if output_path is None:
        if experiment:
            output_path = str(env.submissions_dir / f"{experiment}_submission.csv")
        else:
            output_path = str(env.submissions_dir / "submission.csv")

    click.echo(f"Input: {input_path}")
    df = pd.read_csv(input_path)

    # Internal standard format: sample_name, predicted_label, probabilities
    submission_cols = ["sample_name", "predicted_label"]

    if "predicted_emotion" not in df.columns:
        df["predicted_emotion"] = df["predicted_label"].map(IDX_TO_EMOTION)
    submission_cols.append("predicted_emotion")

    if include_probabilities:
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        submission_cols.extend(prob_cols)

    submission = df[submission_cols].copy()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)

    click.echo(f"Submission saved: {output} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
