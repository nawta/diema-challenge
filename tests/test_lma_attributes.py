"""Unit tests for diema/features/lma_attributes.py (exp087)."""

from __future__ import annotations

import numpy as np
import pytest

from diema.features.lma_attributes import (
    LMA_NAMES,
    compute_batch,
    compute_lma_attributes,
)


def _identity_quats(F: int) -> np.ndarray:
    q = np.zeros((F, 24, 4), dtype=np.float32)
    q[:, :, 0] = 1.0
    return q


def test_schema_is_32():
    assert len(LMA_NAMES) == 32
    assert len(set(LMA_NAMES)) == 32
    for axis in ("body.", "effort.", "shape.", "space."):
        assert sum(n.startswith(axis) for n in LMA_NAMES) == 8


def test_shape_dtype_finite():
    F = 100
    rng = np.random.default_rng(0)
    rp = rng.normal(size=(F, 3)).astype(np.float32)
    q = rng.normal(size=(F, 24, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    out = compute_lma_attributes(rp, q)
    assert out.shape == (32,)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_static_clip_low_motion_features():
    F = 60
    rp = np.zeros((F, 3), dtype=np.float32)
    out = compute_lma_attributes(rp, _identity_quats(F))
    names = {n: i for i, n in enumerate(LMA_NAMES)}
    # No motion → zero speed-derived features
    assert out[names["effort.intensity_peak"]] == pytest.approx(0.0, abs=1e-5)
    assert out[names["effort.intensity_var"]] == pytest.approx(0.0, abs=1e-5)
    assert out[names["space.root_xy_disp"]] == pytest.approx(0.0, abs=1e-5)
    assert np.isfinite(out).all()


def test_translation_increases_displacement():
    F = 50
    rp_still = np.zeros((F, 3), dtype=np.float32)
    rp_move = np.zeros((F, 3), dtype=np.float32)
    rp_move[:, 0] = np.linspace(0, 5, F)  # walk along x
    names = {n: i for i, n in enumerate(LMA_NAMES)}
    s = compute_lma_attributes(rp_still, _identity_quats(F))
    m = compute_lma_attributes(rp_move, _identity_quats(F))
    assert m[names["space.root_xy_disp"]] > s[names["space.root_xy_disp"]]
    assert m[names["space.directness"]] > s[names["space.directness"]] - 1e-6


def test_short_clip_no_crash():
    for F in (1, 2, 3):
        out = compute_lma_attributes(
            np.zeros((F, 3), dtype=np.float32), _identity_quats(F)
        )
        assert out.shape == (32,)
        assert np.isfinite(out).all()


def test_batch():
    F = 40
    rp = [np.zeros((F, 3), dtype=np.float32) for _ in range(4)]
    q = [_identity_quats(F) for _ in range(4)]
    out = compute_batch(rp, q)
    assert out.shape == (4, 32)
    assert np.isfinite(out).all()


def test_invalid_shapes_raise():
    with pytest.raises(ValueError):
        compute_lma_attributes(np.zeros((10, 2)), _identity_quats(10))
    with pytest.raises(ValueError):
        compute_lma_attributes(np.zeros((10, 3)), np.zeros((10, 25, 4)))
    with pytest.raises(ValueError):
        compute_lma_attributes(np.zeros((10, 3)), _identity_quats(9))
