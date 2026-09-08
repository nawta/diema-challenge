"""Unit tests for diema/features/bvh_to_coco17_heatmap.py (exp086)."""

from __future__ import annotations

import numpy as np
import pytest

from diema.features.bvh_to_coco17_heatmap import (
    COCO17_TO_BVH,
    _uniform_sample,
    project_bvh_to_coco17_heatmap,
)


def _identity_quats(F: int) -> np.ndarray:
    q = np.zeros((F, 24, 4), dtype=np.float32)
    q[:, :, 0] = 1.0
    return q


def test_mapping_covers_17():
    assert sorted(COCO17_TO_BVH.keys()) == list(range(17))


def test_shape_and_dtype():
    F = 100
    out = project_bvh_to_coco17_heatmap(
        np.random.randn(F, 3).astype(np.float32), _identity_quats(F),
        clip_len=48, hw=64,
    )
    assert out.shape == (17, 48, 64, 64)
    assert out.dtype == np.float32


def test_heatmap_value_range():
    F = 40
    rng = np.random.default_rng(0)
    rp = rng.normal(size=(F, 3)).astype(np.float32)
    q = rng.normal(size=(F, 24, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    out = project_bvh_to_coco17_heatmap(rp, q)
    # Gaussian peak * confidence ∈ (0, 1]; non-negative
    assert out.min() >= 0.0
    assert out.max() <= 1.0 + 1e-5


def test_body_channels_have_signal():
    F = 30
    rng = np.random.default_rng(1)
    rp = rng.normal(size=(F, 3)).astype(np.float32)
    q = rng.normal(size=(F, 24, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    out = project_bvh_to_coco17_heatmap(rp, q)
    # body keypoints (shoulders..ankles = 5..16) should have non-trivial mass
    body_mass = out[5:17].sum()
    assert body_mass > 0.0


def test_face_channels_lower_than_body():
    F = 30
    rng = np.random.default_rng(2)
    rp = rng.normal(size=(F, 3)).astype(np.float32)
    q = rng.normal(size=(F, 24, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    out = project_bvh_to_coco17_heatmap(rp, q)
    # face joints (0..4) synthesized with conf ≤ 0.5 → peak ≤ 0.5
    assert out[1:5].max() <= 0.3 + 1e-5
    assert out[0].max() <= 0.5 + 1e-5


def test_uniform_sample():
    idx = _uniform_sample(443, 48)
    assert idx.shape == (48,)
    assert idx[0] == 0 and idx[-1] == 442
    assert np.all(np.diff(idx) >= 0)
    idx1 = _uniform_sample(1, 48)
    assert np.all(idx1 == 0)


def test_invalid_shapes_raise():
    with pytest.raises(ValueError):
        project_bvh_to_coco17_heatmap(np.zeros((10, 2)), _identity_quats(10))
    with pytest.raises(ValueError):
        project_bvh_to_coco17_heatmap(np.zeros((10, 3)), np.zeros((10, 17, 4)))
    with pytest.raises(ValueError):
        project_bvh_to_coco17_heatmap(
            np.zeros((10, 3)), _identity_quats(10), drop_axis="w"
        )
