"""Unit tests for diema/features/bvh_to_h36m17.py — exp081 projector."""

from __future__ import annotations

import numpy as np
import pytest

from diema.features.bvh_to_h36m17 import (
    H36M_TO_BVH,
    _uniform_stride_indices,
    project_batch,
    project_bvh_to_h36m17,
)


def _identity_quats(F: int) -> np.ndarray:
    q = np.zeros((F, 24, 4), dtype=np.float32)
    q[:, :, 0] = 1.0
    return q


def test_shape_and_dtype():
    F = 100
    out = project_bvh_to_h36m17(
        np.zeros((F, 3), dtype=np.float32),
        _identity_quats(F),
        target_frames=243,
    )
    assert out.shape == (243, 17, 3)
    assert out.dtype == np.float32


def test_root_is_zero_after_root_relative():
    F = 60
    rng = np.random.default_rng(seed=0)
    rp = rng.normal(size=(F, 3)).astype(np.float32)
    out = project_bvh_to_h36m17(rp, _identity_quats(F))
    np.testing.assert_array_almost_equal(out[:, 0, 0:2], 0.0, decimal=5)


def test_confidence_pattern():
    F = 60
    rng = np.random.default_rng(seed=0)
    rp = rng.normal(size=(F, 3)).astype(np.float32)
    out = project_bvh_to_h36m17(rp, _identity_quats(F))
    body_idx = [i for i, v in H36M_TO_BVH.items() if isinstance(v, int) or v in ("root_synth", "belly_synth")]
    nose_idx = 9
    assert (out[:, body_idx, 2] == 1.0).all()
    assert (out[:, nose_idx, 2] == 0.5).all()


def test_scale_in_unit_range():
    F = 100
    rng = np.random.default_rng(seed=0)
    rp = rng.normal(scale=20.0, size=(F, 3)).astype(np.float32)
    quats = rng.normal(size=(F, 24, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    out = project_bvh_to_h36m17(rp, quats, scale_to_unit=True)
    assert np.abs(out[:, :, :2]).max() <= 1.0 + 1e-5


def test_deterministic():
    F = 64
    rng = np.random.default_rng(seed=7)
    rp = rng.normal(size=(F, 3)).astype(np.float32)
    q = rng.normal(size=(F, 24, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    a = project_bvh_to_h36m17(rp, q)
    b = project_bvh_to_h36m17(rp, q)
    np.testing.assert_array_equal(a, b)


def test_h36m_layout_relations():
    """Sanity check that head should be above hip in normalized image-y."""
    F = 32
    rng = np.random.default_rng(seed=11)
    rp = rng.normal(scale=5.0, size=(F, 3)).astype(np.float32)
    quats = _identity_quats(F)
    out = project_bvh_to_h36m17(rp, quats, flip_vertical=True)
    head_y = out[:, 10, 1].mean()
    rhip_y = out[:, 1, 1].mean()
    lhip_y = out[:, 4, 1].mean()
    hip_y = 0.5 * (rhip_y + lhip_y)
    # screen-y convention: head smaller y (top of frame), hip larger y (bottom)
    assert head_y < hip_y, f"head_y {head_y} should be < hip_y {hip_y} after flip"


def test_batch_stacks():
    F = 40
    rp_list = [np.zeros((F, 3), dtype=np.float32) for _ in range(3)]
    q_list = [_identity_quats(F) for _ in range(3)]
    out = project_batch(rp_list, q_list, target_frames=243)
    assert out.shape == (3, 243, 17, 3)


def test_rejects_wrong_shapes():
    with pytest.raises(ValueError, match="root_pos must be"):
        project_bvh_to_h36m17(np.zeros((10, 4), dtype=np.float32), _identity_quats(10))
    with pytest.raises(ValueError, match="joint_quats must be"):
        project_bvh_to_h36m17(np.zeros((10, 3), dtype=np.float32), np.zeros((10, 25, 4), dtype=np.float32))


def test_uniform_stride():
    idx = _uniform_stride_indices(100, 16)
    assert idx[0] == 0 and idx[-1] == 99 and (np.diff(idx) > 0).all()
