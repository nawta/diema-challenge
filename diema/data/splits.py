"""Leave-Performer-Out (LPO) cross-validation split generation.

Generates K-fold splits where each fold holds out a group of performers
for validation/test, using the rest for training. Splits are deterministic:
sorted performers are distributed round-robin into K groups.

See: diema_challenge_implementation_spec.md §6.2 — split生成
Related: diema/data/parser.py (performer_id extraction), tools/build_splits.py

Ported from: an internal baseline with additions:
  - save/load functionality
  - validation helpers
  - metadata tracking
"""

from collections import defaultdict
from pathlib import Path

from diema.data.parser import parse_performer_id


def generate_lpo_splits(
    filenames: list[str],
    num_folds: int = 10,
    actor_fn=parse_performer_id,
) -> list[dict]:
    """Generate K-fold Leave-Performer-Out splits from a list of filenames.

    Steps:
      1. Extract performer IDs from filenames via actor_fn
      2. Sort unique performers alphabetically (deterministic)
      3. Distribute performers round-robin into num_folds groups
      4. For each fold: held-out group -> val and test, rest -> train

    Round-robin ensures balanced folds. With 92 performers and 10 folds,
    folds 0-1 get 10 performers each, folds 2-9 get 9 each.

    Args:
        filenames: list of filename stems (one per clip in the dataset)
        num_folds: number of folds (K), must be >= 2
        actor_fn: callable that extracts a performer ID from a filename stem.
            Defaults to parse_performer_id (DIEM-A convention).

    Returns:
        List of K split dicts. Each dict has keys 'train', 'val', 'test'.
        Each value is a list of (filename, original_index) tuples.
    """
    if num_folds < 2:
        raise ValueError(f"num_folds must be >= 2, got {num_folds}")

    # Group clip indices by performer
    performer_to_indices = defaultdict(list)
    for idx, fname in enumerate(filenames):
        performer_id = actor_fn(fname)
        performer_to_indices[performer_id].append(idx)

    sorted_performers = sorted(performer_to_indices.keys())

    if num_folds > len(sorted_performers):
        raise ValueError(
            f"num_folds ({num_folds}) exceeds number of unique performers "
            f"({len(sorted_performers)})"
        )

    # Round-robin assignment: performer i -> fold (i % num_folds)
    fold_performers = [[] for _ in range(num_folds)]
    for i, performer in enumerate(sorted_performers):
        fold_performers[i % num_folds].append(performer)

    # Build split dicts
    splits = []
    for fold_idx in range(num_folds):
        holdout_performers = set(fold_performers[fold_idx])
        train_entries = []
        val_entries = []

        for idx, fname in enumerate(filenames):
            performer_id = actor_fn(fname)
            entry = (fname, idx)
            if performer_id in holdout_performers:
                val_entries.append(entry)
            else:
                train_entries.append(entry)

        splits.append({
            "train": train_entries,
            "val": val_entries,
            "test": val_entries,  # val == test for LPO
        })

    return splits


def get_split_metadata(splits: list[dict]) -> dict:
    """Compute summary metadata for a set of splits.

    Args:
        splits: list of split dicts from generate_lpo_splits

    Returns:
        Dict with fold-level counts and performer info.
    """
    metadata = {
        "num_folds": len(splits),
        "folds": [],
    }
    for fold_idx, split in enumerate(splits):
        metadata["folds"].append({
            "fold": fold_idx,
            "train_count": len(split["train"]),
            "val_count": len(split["val"]),
        })
    return metadata


def validate_no_leakage(
    splits: list[dict],
    actor_fn=parse_performer_id,
) -> bool:
    """Verify that no performer appears in both train and val for any fold.

    Args:
        splits: list of split dicts from generate_lpo_splits
        actor_fn: callable to extract performer ID

    Returns:
        True if no leakage detected.

    Raises:
        AssertionError: if any fold has performer overlap between train/val.
    """
    for fold_idx, split in enumerate(splits):
        train_performers = {actor_fn(fname) for fname, _ in split["train"]}
        val_performers = {actor_fn(fname) for fname, _ in split["val"]}
        overlap = train_performers & val_performers
        assert len(overlap) == 0, (
            f"Fold {fold_idx}: performer leakage detected! "
            f"Overlap: {overlap}"
        )
    return True


def save_splits(splits: list[dict], path: str | Path) -> None:
    """Save splits to pickle file."""
    import pickle
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(splits, f)


def load_splits(path: str | Path) -> list[dict]:
    """Load splits from pickle file."""
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
