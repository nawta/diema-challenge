"""Tests for LPO split generation — no performer leakage.

See: diema/data/splits.py
"""

import pytest

from diema.data.parser import parse_performer_id
from diema.data.splits import generate_lpo_splits, validate_no_leakage, get_split_metadata


class TestGenerateLpoSplits:
    """Test LPO split generation logic."""

    def test_basic_split(self, sample_filenames):
        splits = generate_lpo_splits(sample_filenames, num_folds=3)
        assert len(splits) == 3
        for split in splits:
            assert "train" in split
            assert "val" in split
            assert "test" in split

    def test_no_performer_leakage(self, sample_filenames):
        """Critical: no performer should appear in both train and val."""
        splits = generate_lpo_splits(sample_filenames, num_folds=3)
        for fold_idx, split in enumerate(splits):
            train_performers = {parse_performer_id(f) for f, _ in split["train"]}
            val_performers = {parse_performer_id(f) for f, _ in split["val"]}
            overlap = train_performers & val_performers
            assert len(overlap) == 0, (
                f"Fold {fold_idx}: performer leakage! Overlap: {overlap}"
            )

    def test_validate_no_leakage_passes(self, sample_filenames):
        splits = generate_lpo_splits(sample_filenames, num_folds=3)
        assert validate_no_leakage(splits) is True

    def test_all_samples_covered(self, sample_filenames):
        """Every sample must appear in exactly one fold's val set."""
        splits = generate_lpo_splits(sample_filenames, num_folds=3)
        all_val_indices = set()
        for split in splits:
            val_indices = {idx for _, idx in split["val"]}
            # No overlap between folds
            assert len(all_val_indices & val_indices) == 0
            all_val_indices |= val_indices
        # All samples covered
        assert all_val_indices == set(range(len(sample_filenames)))

    def test_train_val_complementary(self, sample_filenames):
        """Train + val should cover all samples in each fold."""
        splits = generate_lpo_splits(sample_filenames, num_folds=3)
        for split in splits:
            train_indices = {idx for _, idx in split["train"]}
            val_indices = {idx for _, idx in split["val"]}
            assert train_indices | val_indices == set(range(len(sample_filenames)))
            assert len(train_indices & val_indices) == 0

    def test_deterministic(self, sample_filenames):
        """Same input should always produce same splits."""
        splits1 = generate_lpo_splits(sample_filenames, num_folds=3)
        splits2 = generate_lpo_splits(sample_filenames, num_folds=3)
        for s1, s2 in zip(splits1, splits2):
            assert s1["train"] == s2["train"]
            assert s1["val"] == s2["val"]

    def test_val_equals_test(self, sample_filenames):
        """In LPO, val == test."""
        splits = generate_lpo_splits(sample_filenames, num_folds=3)
        for split in splits:
            assert split["val"] == split["test"]

    def test_min_folds_validation(self, sample_filenames):
        with pytest.raises(ValueError, match="num_folds must be >= 2"):
            generate_lpo_splits(sample_filenames, num_folds=1)

    def test_too_many_folds(self, sample_filenames):
        """More folds than performers should raise."""
        with pytest.raises(ValueError, match="exceeds"):
            generate_lpo_splits(sample_filenames, num_folds=100)


class TestSplitMetadata:
    def test_metadata_structure(self, sample_filenames):
        splits = generate_lpo_splits(sample_filenames, num_folds=3)
        meta = get_split_metadata(splits)
        assert meta["num_folds"] == 3
        assert len(meta["folds"]) == 3
        for fold_info in meta["folds"]:
            assert "train_count" in fold_info
            assert "val_count" in fold_info
            assert fold_info["train_count"] + fold_info["val_count"] == len(sample_filenames)
