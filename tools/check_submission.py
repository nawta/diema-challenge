"""Validate submission file format.

Usage:
    python tools/check_submission.py --file output/submissions/xxx.csv

See: diema_challenge_implementation_spec.md §6.8, Makefile check-submission
"""

import click
import pandas as pd
from pathlib import Path

from diema.data.parser import NUM_CLASSES


@click.command()
@click.option("--file", "file_path", required=True, help="Submission CSV path")
@click.option("--test-csv", type=str, default=None, help="test_data.csv for sample_name validation")
@click.option("--expected-count", type=int, default=1944, help="Expected number of test samples")
def main(file_path: str, test_csv: str, expected_count: int):
    """Validate a submission file."""
    errors = []
    warnings = []

    path = Path(file_path)
    if not path.exists():
        click.echo(f"ERROR: File not found: {path}", err=True)
        raise SystemExit(1)

    df = pd.read_csv(path)
    click.echo(f"Loaded: {path} ({len(df)} rows, {len(df.columns)} columns)")

    # Check required columns
    if "sample_name" not in df.columns:
        errors.append("Missing required column: sample_name")
    if "predicted_label" not in df.columns:
        errors.append("Missing required column: predicted_label")

    if errors:
        for e in errors:
            click.echo(f"  ERROR: {e}", err=True)
        raise SystemExit(1)

    # Check row count
    if len(df) != expected_count:
        warnings.append(f"Row count {len(df)} != expected {expected_count}")

    # Check duplicates
    dups = df["sample_name"].duplicated().sum()
    if dups > 0:
        errors.append(f"Duplicate sample_names: {dups}")

    # Check label range
    labels = df["predicted_label"].values
    out_of_range = ((labels < 0) | (labels >= NUM_CLASSES)).sum()
    if out_of_range > 0:
        errors.append(f"Labels out of range [0, {NUM_CLASSES - 1}]: {out_of_range}")

    # Check against test_data.csv if provided
    if test_csv is not None:
        test_df = pd.read_csv(test_csv)
        if "sample_name" in test_df.columns:
            expected_names = set(test_df["sample_name"])
            actual_names = set(df["sample_name"])
            missing = expected_names - actual_names
            extra = actual_names - expected_names
            if missing:
                errors.append(f"Missing {len(missing)} sample_names from test_data.csv")
            if extra:
                warnings.append(f"Extra {len(extra)} sample_names not in test_data.csv")

    # Report
    if errors:
        click.echo("\nERRORS:")
        for e in errors:
            click.echo(f"  - {e}", err=True)
    if warnings:
        click.echo("\nWARNINGS:")
        for w in warnings:
            click.echo(f"  - {w}")

    if not errors:
        click.echo("\nSubmission PASSED all checks.")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
