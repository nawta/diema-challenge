"""/2 tests — C3D parser + stats on synthetic and real data.

Related:
- diema/data/c3d_parser.py
- diema/features/c3d_stats.py

Tests skip cleanly when no real DIEM-A C3D file is present; synthetic tests
are always run to verify math.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diema.data.c3d_parser import (
    VICON_PRODUCTION_MARKERS,
    VICON_PRODUCTION_PART_GROUPS,
    group_markers_by_part,
    read_c3d,
)
from diema.features.c3d_stats import (
    compute_c3d_stats,
    feature_names,
    stats_to_vector,
)


REAL_C3D = Path("data/diema_challenge/raw/c3d/train/JP_06_anger_1_H.c3d")
real_available = REAL_C3D.exists()


# ---------------------- Group + marker layout invariants ----------------------


def test_vicon_production_57_markers_total() -> None:
    """The authoritative marker list must sum to exactly 57 across the 7 parts."""
    counts = [len(v) for v in VICON_PRODUCTION_PART_GROUPS.values()]
    assert sum(counts) == 57
    assert len(VICON_PRODUCTION_MARKERS) == 57


def test_part_groups_are_disjoint() -> None:
    """Every marker name belongs to at most one part."""
    seen: set[str] = set()
    for names in VICON_PRODUCTION_PART_GROUPS.values():
        for n in names:
            assert n not in seen, f"duplicate marker {n}"
            seen.add(n)


def test_group_markers_by_part_round_trip() -> None:
    groups = group_markers_by_part(VICON_PRODUCTION_MARKERS)
    for part, idx in groups.items():
        names_back = [VICON_PRODUCTION_MARKERS[i] for i in idx]
        assert names_back == VICON_PRODUCTION_PART_GROUPS[part]


def test_group_markers_tolerates_unknown() -> None:
    """Extra markers in the label list should not crash — just get ignored."""
    labels = VICON_PRODUCTION_MARKERS + ["NEW_MARKER"]
    groups = group_markers_by_part(labels)
    assert sum(len(v) for v in groups.values()) == 57


# ------------------------- Synthetic stats math ------------------------------


def _synthetic_markers(F: int = 60, seed: int = 0) -> np.ndarray:
    """Build a (F, 57, 3) synthetic trajectory with known properties."""
    rng = np.random.default_rng(seed)
    # Start from a canonical standing pose (mm units)
    base = rng.normal(scale=50.0, size=(57, 3))
    base[:, 2] += 1000.0  # place it roughly at 1m height
    traj = np.broadcast_to(base, (F, 57, 3)).copy()
    # Add a gentle oscillation along X to make motion stats non-trivial
    t = np.arange(F)[:, None, None]
    traj += 10.0 * np.sin(t * 0.1)
    return traj.astype(np.float32)


def test_stats_returns_stable_keys() -> None:
    markers = _synthetic_markers()
    stats = compute_c3d_stats(markers, VICON_PRODUCTION_MARKERS)
    names = feature_names()
    assert set(names) == set(stats.keys()), \
        "feature_names() must cover every key produced by compute_c3d_stats"


def test_stats_stationary_clip_has_zero_speed() -> None:
    """A fully stationary trajectory should report ~0 speed/accel/jerk."""
    markers = np.zeros((30, 57, 3), dtype=np.float32)
    markers += 1.0  # non-zero position but no motion
    stats = compute_c3d_stats(markers, VICON_PRODUCTION_MARKERS, rate=120.0)
    assert stats["global_mean_speed"] == 0.0
    assert stats["global_max_speed"] == 0.0
    assert stats["global_path_length"] == 0.0
    assert stats["arm_speed_asym"] == 0.0


def test_stats_respects_rate_for_speed_units() -> None:
    """Speed = frame-diff * Hz; doubling Hz should double reported speed."""
    markers = _synthetic_markers()
    a = compute_c3d_stats(markers, VICON_PRODUCTION_MARKERS, rate=60.0)
    b = compute_c3d_stats(markers, VICON_PRODUCTION_MARKERS, rate=120.0)
    # 2× rate → 2× speed scalar
    assert b["global_mean_speed"] == pytest.approx(a["global_mean_speed"] * 2.0, rel=1e-5)


def test_stats_dropout_counts_negative_residual() -> None:
    markers = _synthetic_markers(F=30)
    residual = np.zeros((30, 57), dtype=np.float32)
    # mark 10% of samples as dropped
    residual[::10] = -1.0
    stats = compute_c3d_stats(
        markers, VICON_PRODUCTION_MARKERS, rate=120.0, residual=residual,
    )
    assert stats["marker_dropout_rate"] == pytest.approx(3 / 30, abs=0.01)


def test_stats_to_vector_preserves_ordering_and_size() -> None:
    markers = _synthetic_markers()
    stats = compute_c3d_stats(markers, VICON_PRODUCTION_MARKERS)
    vec = stats_to_vector(stats)
    assert vec.shape == (len(feature_names()),)
    # Reconstruct first entry from its name
    first_key = feature_names()[0]
    assert vec[0] == pytest.approx(stats[first_key])


def test_stats_to_vector_zero_fills_missing_keys() -> None:
    vec = stats_to_vector({"global_path_length": 123.0})
    names = feature_names()
    assert vec[names.index("global_path_length")] == pytest.approx(123.0)
    # All other entries should be zero
    others = [v for i, v in enumerate(vec) if names[i] != "global_path_length"]
    assert all(o == 0.0 for o in others)


def test_stats_raises_on_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"markers must be"):
        compute_c3d_stats(np.zeros((30, 57), dtype=np.float32), VICON_PRODUCTION_MARKERS)


# ---------------------------- Real-data smoke --------------------------------


@pytest.mark.skipif(not real_available, reason="no DIEM-A C3D data available")
def test_read_c3d_strips_subject_prefix() -> None:
    d = read_c3d(REAL_C3D)
    assert d["n_markers"] == 57
    assert d["rate"] == 120.0
    assert d["subject"] == "JP_06"
    # marker_labels must NOT contain the "JP_06:" prefix
    for lab in d["marker_labels"]:
        assert ":" not in lab, f"prefix leaked through: {lab}"


@pytest.mark.skipif(not real_available, reason="no DIEM-A C3D data available")
def test_read_c3d_respects_strict_flag() -> None:
    d = read_c3d(REAL_C3D, strict=True)
    assert d["n_markers"] == 57


@pytest.mark.skipif(not real_available, reason="no DIEM-A C3D data available")
def test_compute_stats_on_real_clip() -> None:
    d = read_c3d(REAL_C3D)
    stats = compute_c3d_stats(
        d["markers"], d["marker_labels"], rate=d["rate"], residual=d["residual"],
    )
    assert set(stats.keys()) == set(feature_names())
    # JP_06 anger_1_H is 443 frames at 120 Hz ≈ 3.69 s
    assert stats["duration_sec"] == pytest.approx(d["n_frames"] / d["rate"])
    # All speeds / paths should be non-negative on real data
    for k, v in stats.items():
        if "speed" in k or "path" in k or "rms" in k or "sway" in k:
            assert v >= 0.0, f"{k} should be non-negative, got {v}"
