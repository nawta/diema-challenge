"""Unit tests for diema/features/bvh_to_ntu25.py — exp085 MAMP projector."""

from __future__ import annotations

import numpy as np
import pytest

from diema.features.bvh_to_ntu25 import (
    NTU25_TO_BVH,
    _uniform_stride_indices,
    project_batch,
    project_bvh_to_ntu25,
)


def _identity_quats(F: int) -> np.ndarray:
    q = np.zeros((F, 24, 4), dtype=np.float32)
    q[:, :, 0] = 1.0  # wxyz identity
    return q


def test_shape_and_dtype():
    F = 100
    out = project_bvh_to_ntu25(
        np.zeros((F, 3), dtype=np.float32), _identity_quats(F), target_frames=120
    )
    assert out.shape == (120, 25, 3)
    assert out.dtype == np.float32


def test_mapping_covers_all_25_joints():
    assert sorted(NTU25_TO_BVH.keys()) == list(range(25))
    assert all(0 <= v <= 23 for v in NTU25_TO_BVH.values())


def test_spinemid_is_origin_when_unit():
    F = 60
    rng = np.random.default_rng(seed=0)
    rp = rng.normal(size=(F, 3)).astype(np.float32)
    out = project_bvh_to_ntu25(rp, _identity_quats(F), normalize="unit")
    # NTU SpineMid (idx 1) is the centering joint → exactly zero
    np.testing.assert_array_almost_equal(out[:, 1, :], 0.0, decimal=5)


def test_spinemid_is_origin_when_center():
    F = 40
    rng = np.random.default_rng(seed=1)
    rp = rng.normal(size=(F, 3)).astype(np.float32)
    out = project_bvh_to_ntu25(rp, _identity_quats(F), normalize="center")
    np.testing.assert_array_almost_equal(out[:, 1, :], 0.0, decimal=5)


def test_unit_scale_in_range():
    F = 50
    rng = np.random.default_rng(seed=2)
    rp = (rng.normal(size=(F, 3)) * 100).astype(np.float32)
    q = rng.normal(size=(F, 24, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    out = project_bvh_to_ntu25(rp, q, normalize="unit")
    assert np.abs(out).max() <= 1.0 + 1e-5


def test_none_normalize_is_raw_fk():
    F = 30
    rng = np.random.default_rng(seed=3)
    rp = rng.normal(size=(F, 3)).astype(np.float32)
    # target_frames == F → no resampling, SpineBase (idx 0) ← BVH Hips ←
    # root_pos under identity rotation
    out = project_bvh_to_ntu25(rp, _identity_quats(F), target_frames=F, normalize="none")
    np.testing.assert_array_almost_equal(out[:, 0, :], rp, decimal=4)


def test_hand_tip_thumb_duplicate_hand():
    F = 20
    rng = np.random.default_rng(seed=4)
    rp = rng.normal(size=(F, 3)).astype(np.float32)
    q = rng.normal(size=(F, 24, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    out = project_bvh_to_ntu25(rp, q, normalize="none")
    # Left: wrist(6)=hand(7)=tip(21)=thumb(22) all ← BVH 15
    np.testing.assert_array_equal(out[:, 6, :], out[:, 7, :])
    np.testing.assert_array_equal(out[:, 6, :], out[:, 21, :])
    np.testing.assert_array_equal(out[:, 6, :], out[:, 22, :])
    # Right: wrist(10)=hand(11)=tip(23)=thumb(24) all ← BVH 11
    np.testing.assert_array_equal(out[:, 10, :], out[:, 11, :])
    np.testing.assert_array_equal(out[:, 10, :], out[:, 23, :])
    np.testing.assert_array_equal(out[:, 10, :], out[:, 24, :])


def test_uniform_stride_indices():
    idx = _uniform_stride_indices(443, 120)
    assert idx.shape == (120,)
    assert idx[0] == 0
    assert idx[-1] == 442
    assert np.all(np.diff(idx) >= 0)


def test_uniform_stride_single_frame():
    idx = _uniform_stride_indices(1, 120)
    assert idx.shape == (120,)
    assert np.all(idx == 0)


def test_project_batch():
    F = 64
    rp = [np.zeros((F, 3), dtype=np.float32) for _ in range(3)]
    q = [_identity_quats(F) for _ in range(3)]
    out = project_batch(rp, q, target_frames=120)
    assert out.shape == (3, 120, 25, 3)


def test_invalid_shapes_raise():
    with pytest.raises(ValueError):
        project_bvh_to_ntu25(np.zeros((10, 2)), _identity_quats(10))
    with pytest.raises(ValueError):
        project_bvh_to_ntu25(np.zeros((10, 3)), np.zeros((10, 25, 4)))
    with pytest.raises(ValueError):
        project_bvh_to_ntu25(np.zeros((10, 3)), _identity_quats(9))
    with pytest.raises(ValueError):
        project_bvh_to_ntu25(
            np.zeros((10, 3)), _identity_quats(10), normalize="bogus"
        )
