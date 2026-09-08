"""Unit tests for diema/features/bvh_to_halpe14.py — exp077 projector.

Covers:
  - shape contract (3, T, 14)
  - determinism (same input → byte-identical output)
  - confidence-channel pattern (body=1.0, face=0.0 by default)
  - face joints have NaN-free zero coordinates
  - AIDE pixel-scale match (x mean ~505, y mean ~733 across many clips)
  - uniform-stride index correctness
  - body joints have non-degenerate variance (not all zero)
"""

from __future__ import annotations

import numpy as np
import pytest

from diema.features.bvh_to_halpe14 import (
    AIDE_TRAIN_STATS,
    DIEMA_BVH_TO_UBH,
    _uniform_stride_indices,
    project_bvh_to_halpe14,
    project_batch,
)


def _identity_quats(F: int, J: int = 24) -> np.ndarray:
    q = np.zeros((F, J, 4), dtype=np.float32)
    q[:, :, 0] = 1.0
    return q


def test_uniform_stride_indices_handles_short_clip():
    idx = _uniform_stride_indices(num_frames=1, target_frames=16)
    assert idx.shape == (16,)
    assert (idx == 0).all()


def test_uniform_stride_indices_uniform():
    idx = _uniform_stride_indices(num_frames=100, target_frames=16)
    assert idx.shape == (16,)
    assert idx[0] == 0
    assert idx[-1] == 99
    assert (np.diff(idx) > 0).all()


def test_project_shape_and_dtype():
    F = 80
    out = project_bvh_to_halpe14(
        np.zeros((F, 3), dtype=np.float32),
        _identity_quats(F),
        target_frames=16,
    )
    assert out.shape == (3, 16, 14)
    assert out.dtype == np.float32


def test_project_confidence_channel_pattern():
    F = 40
    out = project_bvh_to_halpe14(
        np.zeros((F, 3), dtype=np.float32),
        _identity_quats(F),
    )
    conf = out[2]
    face_idx = [k for k, v in DIEMA_BVH_TO_UBH.items() if v is None]
    body_idx = [k for k, v in DIEMA_BVH_TO_UBH.items() if v is not None]
    assert (conf[:, face_idx] == 0.0).all(), "face joints must be zero-confidence"
    assert (conf[:, body_idx] == 1.0).all(), "body joints must be confidence=1.0"


def test_project_face_joints_default_zero_pad():
    """Default `face_coord_source='zero'` leaves face x,y at 0 with
    confidence=0. (Reverted from 'head' after empirical re-test showed
    head-coord replication hurt every probe variant.)"""
    F = 32
    rng = np.random.default_rng(seed=0)
    quats = rng.normal(size=(F, 24, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    out = project_bvh_to_halpe14(rng.normal(size=(F, 3)).astype(np.float32), quats)
    face_idx = [k for k, v in DIEMA_BVH_TO_UBH.items() if v is None]
    assert (out[2, :, face_idx] == 0.0).all()
    assert (out[0:2, :, face_idx] == 0.0).all()


def test_project_face_joints_head_mode_copies_head_coords():
    """Alternative mode for a review finding evaluation."""
    F = 32
    rng = np.random.default_rng(seed=0)
    quats = rng.normal(size=(F, 24, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    out = project_bvh_to_halpe14(
        rng.normal(size=(F, 3)).astype(np.float32),
        quats,
        face_coord_source="head",
    )
    face_idx = [k for k, v in DIEMA_BVH_TO_UBH.items() if v is None]
    for fi in face_idx:
        np.testing.assert_array_equal(out[0:2, :, fi], out[0:2, :, 0])


def test_project_deterministic_same_input_same_output():
    rng = np.random.default_rng(seed=42)
    F = 60
    rp = rng.normal(size=(F, 3)).astype(np.float32)
    q = rng.normal(size=(F, 24, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    a = project_bvh_to_halpe14(rp, q)
    b = project_bvh_to_halpe14(rp, q)
    np.testing.assert_array_equal(a, b)


def test_project_aide_distribution_match_on_many_clips():
    """Across many random clips the global x/y mean should approach AIDE
    training mean within tolerance (per-clip normalisation centers each
    clip at AIDE mean before scaling)."""
    rng = np.random.default_rng(seed=7)
    F = 50
    n_clips = 20
    body_idx = [k for k, v in DIEMA_BVH_TO_UBH.items() if v is not None]
    xs, ys = [], []
    for _ in range(n_clips):
        rp = rng.normal(scale=10.0, size=(F, 3)).astype(np.float32)
        q = rng.normal(size=(F, 24, 4)).astype(np.float32)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        out = project_bvh_to_halpe14(rp, q)
        xs.append(out[0, :, body_idx].mean())
        ys.append(out[1, :, body_idx].mean())
    assert abs(np.mean(xs) - AIDE_TRAIN_STATS["x"]["mean"]) < 50.0
    assert abs(np.mean(ys) - AIDE_TRAIN_STATS["y"]["mean"]) < 50.0


def test_project_body_joints_have_nontrivial_variance():
    rng = np.random.default_rng(seed=11)
    F = 64
    rp = rng.normal(scale=5.0, size=(F, 3)).astype(np.float32)
    q = rng.normal(size=(F, 24, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    out = project_bvh_to_halpe14(rp, q)
    body_idx = [k for k, v in DIEMA_BVH_TO_UBH.items() if v is not None]
    body_x = out[0, :, body_idx]
    body_y = out[1, :, body_idx]
    assert body_x.std() > 1.0, f"body x variance too small: {body_x.std()}"
    assert body_y.std() > 1.0, f"body y variance too small: {body_y.std()}"


def test_project_batch_stacks_correctly():
    F = 30
    rp_list = [np.zeros((F, 3), dtype=np.float32) for _ in range(3)]
    q_list = [_identity_quats(F) for _ in range(3)]
    batch = project_batch(rp_list, q_list, target_frames=16)
    assert batch.shape == (3, 3, 16, 14, 1)


def test_project_rejects_wrong_shapes():
    with pytest.raises(ValueError, match="root_pos must be"):
        project_bvh_to_halpe14(np.zeros((10, 4), dtype=np.float32), _identity_quats(10))
    with pytest.raises(ValueError, match="joint_quats must be"):
        project_bvh_to_halpe14(np.zeros((10, 3), dtype=np.float32), np.zeros((10, 25, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="root_pos F"):
        project_bvh_to_halpe14(np.zeros((10, 3), dtype=np.float32), _identity_quats(8))


def test_project_y_axis_alternative_drop():
    F = 16
    rng = np.random.default_rng(seed=1)
    rp = rng.normal(size=(F, 3)).astype(np.float32)
    q = _identity_quats(F)
    out_z = project_bvh_to_halpe14(rp, q, drop_axis="z")
    out_y = project_bvh_to_halpe14(rp, q, drop_axis="y")
    assert out_z.shape == out_y.shape
