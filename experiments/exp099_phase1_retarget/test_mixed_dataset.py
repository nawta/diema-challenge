"""Tests for JointPosMixedDataset (Stage 2).

Smoke-level tests; not a full coverage sweep. Verifies:
  - shape / dtype contract
  - mix ratio actually achieved
  - LPO leakage: validation set sees no retargeted clips
  - honest vs flat FK produces different outputs for the same performer
  - reproducible under seed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "exp099_phase1_retarget"))

from diema.data.splits import generate_lpo_splits  # noqa: E402
from mixed_dataset import JointPosMixedDataset  # noqa: E402
import pybvh_ml  # noqa: E402


CACHED_NPZ = "data/diema_challenge/processed/motion_train_quat.npz"


def _load_filenames():
    preprocessed = pybvh_ml.load_preprocessed(CACHED_NPZ)
    return preprocessed["filenames"]


def _fold0_train_set(num_folds: int = 10):
    fnames = _load_filenames()
    splits = generate_lpo_splits(fnames, num_folds=num_folds)
    return splits[0]


def test_shape_and_dtype_contract():
    """Sample from train fold and check (C=3, T, V=25) float32 + label int64."""
    split = _fold0_train_set()
    # Train performers (for retargeted filtering)
    from diema.data.parser import parse_performer_id
    train_performers = {parse_performer_id(f) for f, _ in split["train"]}

    ds = JointPosMixedDataset(
        original_split_entries=split["train"],
        train_performer_ids=train_performers,
        original_fk_rule="flat",
        mix_ratio=0.0,   # No retargeted in this test
        clip_length=64,
        is_test=False,
        augmentation_pipeline=None,
        seed=0,
    )
    x, y, fn = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.dtype == torch.float32
    assert x.shape == (3, 64, 25)
    assert isinstance(y, torch.Tensor)
    assert y.dtype == torch.long
    assert 0 <= y.item() < 12
    assert isinstance(fn, str)


def test_mix_ratio_actually_achieved():
    """With mix_ratio=0.4, retargeted/total should be ≈ 0.4 (±0.02)."""
    split = _fold0_train_set()
    from diema.data.parser import parse_performer_id
    train_performers = {parse_performer_id(f) for f, _ in split["train"]}

    ds = JointPosMixedDataset(
        original_split_entries=split["train"],
        train_performer_ids=train_performers,
        original_fk_rule="flat",
        mix_ratio=0.4,
        clip_length=64,
        is_test=False,
        augmentation_pipeline=None,
        seed=0,
    )
    summary = ds.summary()
    actual = summary["actual_mix_ratio"]
    print(f"  actual mix ratio: {actual:.4f} (target 0.4)")
    print(f"  n_orig={summary['n_original']}, "
          f"n_retgt={summary['n_retargeted_per_epoch']}, "
          f"pool_unique={summary['n_retargeted_pool_unique']}, "
          f"upsampled={summary['upsampled']}")
    # Tight tolerance (≤ 0.01) — upsampling lets us always hit target.
    assert abs(actual - 0.4) < 0.01, (
        f"mix ratio off-target by > 0.01: {actual}")


def test_val_split_has_no_retargeted():
    """Val/test mode forces mix_ratio=0 regardless of arg."""
    split = _fold0_train_set()
    from diema.data.parser import parse_performer_id
    train_performers = {parse_performer_id(f) for f, _ in split["train"]}

    ds = JointPosMixedDataset(
        original_split_entries=split["val"],
        train_performer_ids=train_performers,
        original_fk_rule="flat",
        mix_ratio=0.4,    # arg says 0.4 but is_test=True overrides
        clip_length=64,
        is_test=True,
        augmentation_pipeline=None,
        seed=0,
    )
    summary = ds.summary()
    assert summary["n_retargeted_per_epoch"] == 0
    assert summary["actual_mix_ratio"] == 0.0


def test_lpo_leakage_filter():
    """Retargeted clips from a held-out performer must not appear in train."""
    split = _fold0_train_set()
    from diema.data.parser import parse_performer_id
    train_performers = {parse_performer_id(f) for f, _ in split["train"]}
    val_performers = {parse_performer_id(f) for f, _ in split["val"]}

    ds = JointPosMixedDataset(
        original_split_entries=split["train"],
        train_performer_ids=train_performers,
        original_fk_rule="flat",
        mix_ratio=0.4,
        clip_length=64,
        is_test=False,
        augmentation_pipeline=None,
        seed=0,
    )
    # Walk the eligible retargeted pool: every pair's source performer must
    # be in train set, NOT in val set.
    for p in ds._retargeted_pairs:   # noqa: SLF001 (internal test)
        assert p["performer"] in train_performers, (
            f"Leakage: retargeted source {p['performer']} not in train")
        assert p["performer"] not in val_performers, (
            f"Leakage: retargeted source {p['performer']} found in val")


def test_honest_vs_flat_fk_differ():
    """Same original clip under honest vs flat FK should produce different
    tensors when the source performer is NOT JP_06."""
    split = _fold0_train_set()
    from diema.data.parser import parse_performer_id
    train_performers = {parse_performer_id(f) for f, _ in split["train"]}

    # Find a train clip whose performer is NOT JP_06
    target_idx = None
    for local_i, (fn, _) in enumerate(split["train"]):
        perf = parse_performer_id(fn)
        if perf != "JP_06":
            target_idx = local_i
            break
    assert target_idx is not None, "No non-JP_06 train clip found"

    common = dict(
        original_split_entries=split["train"],
        train_performer_ids=train_performers,
        mix_ratio=0.0,
        clip_length=64,
        is_test=True,    # deterministic temporal sample
        augmentation_pipeline=None,
        seed=0,
    )
    ds_honest = JointPosMixedDataset(original_fk_rule="honest", **common)
    ds_flat = JointPosMixedDataset(original_fk_rule="flat", **common)

    x_honest, y_h, _ = ds_honest[target_idx]
    x_flat, y_f, _ = ds_flat[target_idx]
    # Labels must match
    assert y_h.item() == y_f.item()
    # Joint positions must differ (different bone offsets → different FK)
    diff = (x_honest - x_flat).abs().max().item()
    print(f"  max abs diff honest vs flat: {diff:.4f}")
    assert diff > 0.1, (
        f"honest and flat FK should differ by > 0.1; got max abs diff "
        f"{diff:.6f}")


def test_seed_reproducibility():
    """Same seed → same first sample tensor."""
    split = _fold0_train_set()
    from diema.data.parser import parse_performer_id
    train_performers = {parse_performer_id(f) for f, _ in split["train"]}

    common = dict(
        original_split_entries=split["train"],
        train_performer_ids=train_performers,
        original_fk_rule="flat",
        mix_ratio=0.0,
        clip_length=64,
        is_test=False,
        augmentation_pipeline=None,
    )
    ds_a = JointPosMixedDataset(seed=123, **common)
    ds_b = JointPosMixedDataset(seed=123, **common)
    x_a, y_a, _ = ds_a[0]
    x_b, y_b, _ = ds_b[0]
    assert torch.allclose(x_a, x_b)
    assert y_a.item() == y_b.item()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
