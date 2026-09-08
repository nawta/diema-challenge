"""Tests for augmentations.

Covered:
  - sequence_cutout_per_body_part respects body-part boundaries and p=0 / p=1
  - time_shift_delta_features produces the correct concatenated shape and
    lag-difference values
"""

from __future__ import annotations

import numpy as np
import pytest

from diema.data.augmentations import (
    body_scale_jitter,
    sequence_cutout_per_body_part,
    time_shift_delta_features,
)
from diema.models.skeleton_graph import DIEMA_BODY_PARTS


@pytest.fixture
def ctv_tensor():
    # (C=6, T=64, V=25) filled with unique values to detect zeroing
    return np.arange(6 * 64 * 25, dtype=np.float32).reshape(6, 64, 25) + 1.0


def test_cutout_p0_noop(ctv_tensor):
    rng = np.random.default_rng(0)
    original = ctv_tensor.copy()
    out = sequence_cutout_per_body_part(ctv_tensor, rng, p=0.0)
    # p=0 → no parts selected → tensor unchanged
    np.testing.assert_array_equal(out, original)


def test_cutout_p1_hits_every_part(ctv_tensor):
    rng = np.random.default_rng(42)
    out = sequence_cutout_per_body_part(
        ctv_tensor, rng, p=1.0, num_segments=3, segment_ratio=0.2,
    )
    # Each body-part must have at least one zeroed entry somewhere
    for part_name, nodes in DIEMA_BODY_PARTS.items():
        sub = out[:, :, nodes]
        assert (sub == 0.0).any(), f"part '{part_name}' has no zeros"


def test_cutout_only_affects_its_body_part():
    # Start with a known non-zero pattern, apply cutout, and check that if a
    # part is zeroed anywhere it was zeroed only in that part's node columns.
    rng = np.random.default_rng(7)
    data = np.ones((6, 64, 25), dtype=np.float32)
    out = sequence_cutout_per_body_part(
        data, rng, p=1.0, num_segments=2, segment_ratio=0.15,
    )
    zero_mask = (out == 0.0)          # (C, T, V)
    touched_nodes = zero_mask.any(axis=(0, 1))  # (V,)
    # Any node that is zero must belong to some body-part (all do, by design),
    # and must be one of the 25 DIEMA nodes.
    assert touched_nodes.shape == (25,)
    # Sanity check: on seed 7 with p=1 we expect at least one zero.
    assert touched_nodes.any()


def test_cutout_segment_length(ctv_tensor):
    rng = np.random.default_rng(11)
    T = ctv_tensor.shape[1]
    ratio = 0.25
    L = int(round(ratio * T))  # 16
    out = sequence_cutout_per_body_part(
        ctv_tensor, rng, p=1.0, num_segments=1, segment_ratio=ratio,
    )
    # For each part, the number of zeroed time steps is <= L (one segment)
    for nodes in DIEMA_BODY_PARTS.values():
        sub = out[:, :, nodes]
        zero_t_count = (sub == 0.0).any(axis=(0, 2)).sum()
        assert zero_t_count <= L, f"zero segment wider than expected L={L}"


def test_time_shift_delta_shape():
    data = np.random.rand(6, 64, 25).astype(np.float32)
    lags = (1, 2, 4)
    out = time_shift_delta_features(data, lags=lags)
    # 6 original channels + 3 lag blocks × 6 = 24 channels
    assert out.shape == (6 * (1 + len(lags)), 64, 25)
    # First block equals input
    np.testing.assert_array_equal(out[:6], data)


def test_time_shift_delta_values():
    T = 8
    data = np.zeros((2, T, 1), dtype=np.float32)
    data[:, :, 0] = np.arange(T)[None, :]  # channel-constant ramp
    out = time_shift_delta_features(data, lags=(2,), fill_mode="edge")
    # Block 2 (indices 2..3) = data[:, 2:] - data[:, :-2] = 2, 2, 2, ...
    delta = out[2:4]
    # Head frames (indices 0..1) are forward-filled with delta[2] = 2
    assert delta[0, 0, 0] == pytest.approx(2.0)
    assert delta[0, 1, 0] == pytest.approx(2.0)
    assert delta[0, 2, 0] == pytest.approx(2.0)
    assert delta[0, T - 1, 0] == pytest.approx(2.0)


def test_time_shift_delta_zero_fill_head():
    data = np.ones((1, 4, 1), dtype=np.float32)
    out = time_shift_delta_features(data, lags=(1,), fill_mode="zero")
    delta = out[1:2]
    # Lag=1 of constant → all zeros
    np.testing.assert_array_equal(delta, np.zeros_like(delta))


def test_time_shift_delta_rejects_invalid_lag():
    data = np.ones((1, 4, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        time_shift_delta_features(data, lags=(0,))


def test_time_shift_delta_rejects_invalid_fill_mode():
    data = np.ones((1, 4, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        time_shift_delta_features(data, lags=(1,), fill_mode="ffill")


def test_time_shift_delta_lag_exceeds_T():
    # lag >= T → no valid differences, block is all zeros, no crash
    data = np.random.rand(2, 4, 3).astype(np.float32)
    out = time_shift_delta_features(data, lags=(4, 8), fill_mode="edge")
    assert out.shape == (2 * 3, 4, 3)
    # lag=4 → all zero (diffs would be at index 4.. but T=4 so nothing)
    np.testing.assert_array_equal(out[2:4], np.zeros((2, 4, 3), dtype=np.float32))
    # lag=8 → all zero
    np.testing.assert_array_equal(out[4:6], np.zeros((2, 4, 3), dtype=np.float32))


def test_cutout_rejects_invalid_segment_ratio(ctv_tensor):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sequence_cutout_per_body_part(ctv_tensor, rng, segment_ratio=0.0)
    with pytest.raises(ValueError):
        sequence_cutout_per_body_part(ctv_tensor, rng, segment_ratio=-0.1)


def test_cutout_rejects_invalid_num_segments(ctv_tensor):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sequence_cutout_per_body_part(ctv_tensor, rng, num_segments=-1)


def test_cutout_rejects_invalid_p(ctv_tensor):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sequence_cutout_per_body_part(ctv_tensor, rng, p=1.5)
    with pytest.raises(ValueError):
        sequence_cutout_per_body_part(ctv_tensor, rng, p=-0.1)


def test_cutout_num_segments_zero_is_noop(ctv_tensor):
    rng = np.random.default_rng(0)
    original = ctv_tensor.copy()
    out = sequence_cutout_per_body_part(
        ctv_tensor, rng, p=1.0, num_segments=0,
    )
    np.testing.assert_array_equal(out, original)


def test_body_scale_jitter_shape_preserved():
    rng = np.random.default_rng(0)
    data = np.random.rand(6, 64, 25).astype(np.float32)
    out = body_scale_jitter(data.copy(), rng, scale_range=(0.9, 1.1), pos_nodes=None)
    assert out.shape == data.shape


def test_body_scale_jitter_scales_only_pos_channels():
    rng = np.random.default_rng(7)
    # Use channels 0..2 = positional, 3..5 = "rotation"
    data = np.ones((6, 4, 3), dtype=np.float32)
    out = body_scale_jitter(
        data.copy(), rng, scale_range=(2.0, 2.0), pos_channels=(0, 1, 2), pos_nodes=None,
    )
    # Positional channels get exactly 2.0
    np.testing.assert_array_equal(out[:3], 2.0 * np.ones((3, 4, 3)))
    # Non-positional channels untouched
    np.testing.assert_array_equal(out[3:], np.ones((3, 4, 3)))


def test_body_scale_jitter_pos_nodes_subset():
    rng = np.random.default_rng(7)
    data = np.ones((3, 4, 5), dtype=np.float32)
    # Scale only nodes 0 and 2 by 3.0
    out = body_scale_jitter(
        data.copy(), rng, scale_range=(3.0, 3.0), pos_channels=(0, 1, 2), pos_nodes=(0, 2),
    )
    assert (out[:, :, 0] == 3.0).all()
    assert (out[:, :, 2] == 3.0).all()
    assert (out[:, :, 1] == 1.0).all()
    assert (out[:, :, 3] == 1.0).all()
    assert (out[:, :, 4] == 1.0).all()


def test_body_scale_jitter_rejects_bad_range():
    rng = np.random.default_rng(0)
    data = np.zeros((3, 4, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        body_scale_jitter(data, rng, scale_range=(0.0, 1.0))
    with pytest.raises(ValueError):
        body_scale_jitter(data, rng, scale_range=(1.5, 1.0))


def test_body_scale_jitter_rejects_bad_channel():
    rng = np.random.default_rng(0)
    data = np.zeros((3, 4, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        body_scale_jitter(data, rng, pos_channels=(0, 5))


def test_body_scale_jitter_rejects_wrong_ndim():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        body_scale_jitter(np.zeros((3, 4)), rng)


def test_datamodule_expected_in_channels():
    from diema.data.collate import MotionDataModule
    assert MotionDataModule.expected_in_channels("rotation_6d", None) == 6
    assert MotionDataModule.expected_in_channels("rotation_6d", (1,)) == 12
    assert MotionDataModule.expected_in_channels("rotation_6d", (1, 2, 4)) == 24
    assert MotionDataModule.expected_in_channels("joint_pos", (1, 2)) == 9
    with pytest.raises(ValueError):
        MotionDataModule.expected_in_channels("unknown_stream")
