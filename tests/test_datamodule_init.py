"""Regression test: MotionDataModule.__init__ must populate train/val/test indices.

There was a silent bug where a refactor moved the split-dict parsing code
after a `return` inside the staticmethod `expected_in_channels`, leaving
`MotionDataModule` instances without `train_indices` / `val_indices`. This
test prevents recurrence.
"""

from __future__ import annotations

import pickle

from diema.data.collate import MotionDataModule


def _make_fake_split_dict():
    # Matches the structure of diema.data.splits output:
    # each list contains (filename, global_index) tuples.
    return {
        "train": [("a.npz", 0), ("b.npz", 1), ("c.npz", 2)],
        "val": [("d.npz", 3), ("e.npz", 4)],
    }


def test_datamodule_init_populates_split_indices(tmp_path):
    split = _make_fake_split_dict()
    dm = MotionDataModule(
        data_path=str(tmp_path / "fake.npz"),
        split_dict=split,
    )
    assert hasattr(dm, "train_indices")
    assert hasattr(dm, "val_indices")
    assert hasattr(dm, "test_indices")
    assert dm.train_indices == [0, 1, 2]
    assert dm.val_indices == [3, 4]
    # No explicit test split → falls back to val
    assert dm.test_indices == [3, 4]


def test_datamodule_init_with_explicit_test_indices(tmp_path):
    split = _make_fake_split_dict()
    split["test"] = [("f.npz", 5), ("g.npz", 6), ("h.npz", 7)]
    dm = MotionDataModule(
        data_path=str(tmp_path / "fake.npz"),
        split_dict=split,
    )
    assert dm.test_indices == [5, 6, 7]


def test_datamodule_init_debug_limits_samples(tmp_path):
    # Build a large split to verify debug truncation
    large_train = [(f"{i}.npz", i) for i in range(200)]
    large_val = [(f"{i}.npz", 200 + i) for i in range(200)]
    dm = MotionDataModule(
        data_path=str(tmp_path / "fake.npz"),
        split_dict={"train": large_train, "val": large_val},
        debug=True,
    )
    assert len(dm.train_indices) == 100
    assert len(dm.val_indices) == 100


def test_datamodule_init_from_pickle(tmp_path):
    split = _make_fake_split_dict()
    split_path = tmp_path / "split.pkl"
    with open(split_path, "wb") as f:
        pickle.dump(split, f)
    dm = MotionDataModule(
        data_path=str(tmp_path / "fake.npz"),
        split_path=str(split_path),
    )
    assert dm.train_indices == [0, 1, 2]
    assert dm.val_indices == [3, 4]


def test_datamodule_requires_split_source():
    import pytest
    with pytest.raises(ValueError, match="split"):
        MotionDataModule(data_path="/tmp/fake.npz")


def test_datamodule_expected_in_channels_still_works():
    # Keep coverage on the staticmethod that was previously swallowing __init__
    assert MotionDataModule.expected_in_channels("rotation_6d", None) == 6
    assert MotionDataModule.expected_in_channels("rotation_6d", (1, 2)) == 18


def test_datamodule_passes_body_scale_jitter_range_to_dataset(tmp_path):
    """ MotionDataModule must pass body_scale_jitter_range through
    to MotionDataset constructor without crashing on init."""
    split = _make_fake_split_dict()
    dm = MotionDataModule(
        data_path=str(tmp_path / "fake.npz"),
        split_dict=split,
        body_scale_jitter_range=(0.9, 1.1),
    )
    assert dm.body_scale_jitter_range == (0.9, 1.1)


def test_datamodule_default_body_scale_jitter_range_is_none(tmp_path):
    split = _make_fake_split_dict()
    dm = MotionDataModule(
        data_path=str(tmp_path / "fake.npz"),
        split_dict=split,
    )
    assert dm.body_scale_jitter_range is None
