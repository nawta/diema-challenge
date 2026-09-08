"""Prepare dataset: raw BVH -> processed NPZ.

Usage:
    python tools/prepare_dataset.py
    python tools/prepare_dataset.py --train-only
    python tools/prepare_dataset.py --test-only

See: diema_challenge_implementation_spec.md §6.1, Makefile prepare-data
"""

import click
from pathlib import Path

from utils.env import EnvConfig


@click.command()
@click.option("--train-only", is_flag=True, help="Process only training data")
@click.option("--test-only", is_flag=True, help="Process only test data")
@click.option("--augment-copies", type=int, default=0, help="Number of augmented copies per sample")
def main(train_only: bool, test_only: bool, augment_copies: int):
    """Preprocess raw BVH files into quaternion NPZ format."""
    from diema.data.preprocess import preprocess_train, preprocess_test, validate_preprocessed

    env = EnvConfig()
    train_bvh_dir = env.raw_dir / "bvh" / "train"
    test_bvh_dir = env.raw_dir / "bvh" / "test"
    train_npz = env.processed_dir / "motion_train_quat.npz"
    test_npz = env.processed_dir / "motion_test_quat.npz"

    if not test_only:
        click.echo(f"Processing train BVH: {train_bvh_dir}")
        if not train_bvh_dir.exists():
            click.echo(f"ERROR: Train BVH directory not found: {train_bvh_dir}", err=True)
            raise SystemExit(1)

        if augment_copies > 0:
            from diema.data.preprocess import preprocess_with_augmentation
            preprocess_with_augmentation(train_bvh_dir, train_npz, augment_copies=augment_copies)
        else:
            preprocess_train(train_bvh_dir, train_npz)

        stats = validate_preprocessed(train_npz)
        click.echo(f"Train: {stats['num_clips']} clips, "
                    f"seq_len [{stats['seq_length_min']}-{stats['seq_length_max']}], "
                    f"mean={stats['seq_length_mean']:.1f}")
        if "num_performers" in stats:
            click.echo(f"  Performers: {stats['num_performers']}")
        if "label_distribution" in stats:
            click.echo(f"  Labels: {stats['label_distribution']}")

    if not train_only:
        click.echo(f"Processing test BVH: {test_bvh_dir}")
        if not test_bvh_dir.exists():
            click.echo(f"ERROR: Test BVH directory not found: {test_bvh_dir}", err=True)
            raise SystemExit(1)

        preprocess_test(test_bvh_dir, test_npz)

        stats = validate_preprocessed(test_npz)
        click.echo(f"Test: {stats['num_clips']} clips, "
                    f"seq_len [{stats['seq_length_min']}-{stats['seq_length_max']}], "
                    f"mean={stats['seq_length_mean']:.1f}")

    click.echo("Done.")


if __name__ == "__main__":
    main()
