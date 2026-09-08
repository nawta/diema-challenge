"""Generate LPO cross-validation splits.

Usage:
    python tools/build_splits.py
    python tools/build_splits.py --num-folds 5

See: diema_challenge_implementation_spec.md §6.2, Makefile build-splits
"""

import click

import pybvh_ml

from utils.env import EnvConfig
from diema.data.splits import generate_lpo_splits, validate_no_leakage, save_splits, get_split_metadata


@click.command()
@click.option("--num-folds", type=int, default=10, help="Number of LPO folds")
@click.option("--output", type=str, default=None, help="Output pickle path (default: processed/split_lpo_{K}fold.pkl)")
def main(num_folds: int, output: str):
    """Generate Leave-Performer-Out cross-validation splits."""
    env = EnvConfig()
    train_npz = env.processed_dir / "motion_train_quat.npz"

    if not train_npz.exists():
        click.echo(f"ERROR: Train NPZ not found: {train_npz}. Run 'make prepare-data' first.", err=True)
        raise SystemExit(1)

    click.echo(f"Loading preprocessed data: {train_npz}")
    preprocessed = pybvh_ml.load_preprocessed(str(train_npz))
    filenames = preprocessed["filenames"]
    click.echo(f"  {len(filenames)} clips loaded")

    click.echo(f"Generating {num_folds}-fold LPO splits...")
    splits = generate_lpo_splits(filenames, num_folds=num_folds)

    click.echo("Validating no performer leakage...")
    validate_no_leakage(splits)
    click.echo("  OK: no leakage detected")

    metadata = get_split_metadata(splits)
    for fold_info in metadata["folds"]:
        click.echo(f"  Fold {fold_info['fold']}: train={fold_info['train_count']}, val={fold_info['val_count']}")

    if output is None:
        output = str(env.processed_dir / f"split_lpo_{num_folds}fold.pkl")

    save_splits(splits, output)
    click.echo(f"Splits saved to {output}")


if __name__ == "__main__":
    main()
