""" per-clip summary statistics from C3D markers.

Related:
- diema/data/c3d_parser.py — source of the marker tensor
- experiments/exp056_c3d_stats_latefusion/ — consumer

Given the (F, 57, 3) marker trajectory and the body-part grouping from
``diema.data.c3d_parser``, compute ~50 scalar summary features per clip:

  - centroid path length + RMS spread
  - left-right asymmetry (per body-region axis)
  - foot height + contact proxy (LTOE/RTOE/LHEL/RHEL vs centroid)
  - root sway (pelvis centroid XY RMS)
  - speed / acceleration / jerk norms (over whole clip + per-part)
  - marker dropout rate (fraction of frames with residual < 0)

No performer identity enters the feature vector — subject prefix is
stripped in the parser. Feature names are stable strings so the
late-fusion head can be wired by name not by position.
"""

from __future__ import annotations

import numpy as np

from diema.data.c3d_parser import (
    VICON_PRODUCTION_PART_GROUPS,
    group_markers_by_part,
)


def _safe_mean(arr: np.ndarray, axis: int | None = None) -> np.ndarray:
    """NaN-safe mean. Treats inf / nan entries as missing."""
    return np.asarray(np.nanmean(arr, axis=axis) if arr.size else 0.0)


def _path_length(points: np.ndarray) -> float:
    """Total Euclidean path length along axis 0 of a (F, 3) trajectory.

    NaN-safe: dropped-marker frames produce NaN, which we treat as zero
    step so the path length is a lower bound rather than NaN.
    """
    if points.shape[0] < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    norms = np.linalg.norm(np.nan_to_num(diffs, nan=0.0), axis=1)
    return float(norms.sum())


def _rms_spread(points: np.ndarray) -> float:
    """Root-mean-square deviation from the mean position.

    NaN-safe via np.nanmean."""
    if points.size == 0:
        return 0.0
    mean = np.nanmean(points, axis=0)
    centered = np.nan_to_num(points - mean, nan=0.0)
    return float(np.sqrt((centered ** 2).sum(axis=1).mean()))


def _temporal_derivative(points: np.ndarray) -> np.ndarray:
    """Row-wise ΔL2 across adjacent frames of a (F, K) tensor. NaN-safe."""
    if points.shape[0] < 2:
        return np.zeros(0, dtype=np.float32)
    diffs = np.nan_to_num(np.diff(points, axis=0), nan=0.0)
    if diffs.ndim == 2:
        return np.linalg.norm(diffs, axis=1)
    return np.linalg.norm(diffs, axis=-1).mean(axis=-1)


def _part_centroid(
    markers: np.ndarray, part_idx: list[int],
) -> np.ndarray:
    """Return the (F, 3) centroid trajectory for a part. Empty part → zeros.
    NaN-safe (dropped markers do not poison the per-part centroid)."""
    if not part_idx:
        return np.zeros((markers.shape[0], 3), dtype=np.float32)
    part_markers = markers[:, part_idx, :]
    centroid = np.nanmean(part_markers, axis=1)
    return np.nan_to_num(centroid, nan=0.0)


def compute_c3d_stats(
    markers: np.ndarray,
    marker_labels: list[str],
    rate: float = 120.0,
    residual: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the full feature dict for one clip.

    Args:
        markers: (F, M, 3) trajectories in mm (ezc3d default).
        marker_labels: labels (subject prefix already stripped).
        rate: sampling rate in Hz. Defaults to DIEM-A's 120 Hz.
        residual: optional (F, M) residual array. Values < 0 ⇒ dropped marker.

    Returns:
        ``dict[str, float]`` with stable keys. The length is fixed if the
        marker set is the canonical Vicon Production 57; missing parts
        silently contribute zeros.
    """
    if markers.ndim != 3 or markers.shape[-1] != 3:
        raise ValueError(f"markers must be (F, M, 3); got {markers.shape}")
    F = markers.shape[0]
    part_idx = group_markers_by_part(marker_labels)

    stats: dict[str, float] = {}

    # --- Global (all-marker) stats --------------------------------------
    # Use nanmean so dropped markers don't poison the centroid trajectory.
    global_centroid = np.nan_to_num(np.nanmean(markers, axis=1), nan=0.0)  # (F, 3)
    stats["global_path_length"] = _path_length(global_centroid)
    stats["global_rms_spread"] = _rms_spread(global_centroid)

    speed = _temporal_derivative(markers) * rate  # mm/s (frame-diff * Hz)
    stats["global_mean_speed"] = float(speed.mean()) if speed.size else 0.0
    stats["global_max_speed"] = float(speed.max()) if speed.size else 0.0

    # Acceleration / jerk norms
    if speed.size >= 2:
        accel = np.abs(np.diff(speed)) * rate
        stats["global_mean_accel"] = float(accel.mean())
        stats["global_max_accel"] = float(accel.max())
    else:
        stats["global_mean_accel"] = 0.0
        stats["global_max_accel"] = 0.0
    if speed.size >= 3:
        jerk = np.abs(np.diff(speed, n=2)) * (rate ** 2)
        stats["global_mean_jerk"] = float(jerk.mean())
    else:
        stats["global_mean_jerk"] = 0.0

    # --- Per-part stats --------------------------------------------------
    for part, idx in part_idx.items():
        centroid = _part_centroid(markers, idx)
        stats[f"{part}_path_length"] = _path_length(centroid)
        stats[f"{part}_rms_spread"] = _rms_spread(centroid)
        part_speed = _temporal_derivative(centroid) * rate
        stats[f"{part}_mean_speed"] = float(part_speed.mean()) if part_speed.size else 0.0
        stats[f"{part}_max_speed"] = float(part_speed.max()) if part_speed.size else 0.0

    # --- Left / right asymmetry -----------------------------------------
    # For arms and legs, compute |mean_speed_L - mean_speed_R|.
    for L, R in (("l_arm", "r_arm"), ("l_leg", "r_leg")):
        l = stats.get(f"{L}_mean_speed", 0.0)
        r = stats.get(f"{R}_mean_speed", 0.0)
        stats[f"{L[:-4]}_arm_speed_asym"] = abs(l - r) if "arm" in L else 0.0
        stats[f"{L[:-4]}_leg_speed_asym"] = abs(l - r) if "leg" in L else 0.0
    # Clean up the placeholders (both loops wrote the same key; only one is real)
    stats["arm_speed_asym"] = abs(
        stats.get("l_arm_mean_speed", 0.0) - stats.get("r_arm_mean_speed", 0.0)
    )
    stats["leg_speed_asym"] = abs(
        stats.get("l_leg_mean_speed", 0.0) - stats.get("r_leg_mean_speed", 0.0)
    )
    # Remove the hyphenated intermediate keys ("_arm_speed_asym" / "_leg_speed_asym")
    for k in (
        "l__arm_speed_asym", "r__arm_speed_asym",
        "l__leg_speed_asym", "r__leg_speed_asym",
    ):
        stats.pop(k, None)

    # --- Foot ground-proxy ----------------------------------------------
    # "Ground" = pelvis centroid Z (height axis is Z in Vicon DIEM-A paper);
    # we use axis=2 as Z. Foot height = toe/heel Z relative to pelvis Z.
    if part_idx.get("pelvis") and part_idx.get("l_leg") and part_idx.get("r_leg"):
        pelvis_z = _part_centroid(markers, part_idx["pelvis"])[:, 2]  # (F,)
        for side, markers_needed in (
            ("l", ("LTOE", "LHEL")),
            ("r", ("RTOE", "RHEL")),
        ):
            for m in markers_needed:
                if m in marker_labels:
                    mid = marker_labels.index(m)
                    z = markers[:, mid, 2]
                    heights = pelvis_z - z  # +ve when foot is below pelvis
                    # NaN-safe: dropped frames get NaN; fall back to 0 / nan-safe reduce.
                    heights_clean = heights[~np.isnan(heights)]
                    if heights_clean.size:
                        stats[f"{m.lower()}_min_height"] = float(heights_clean.min())
                        stats[f"{m.lower()}_mean_height"] = float(heights_clean.mean())
                    else:
                        stats[f"{m.lower()}_min_height"] = 0.0
                        stats[f"{m.lower()}_mean_height"] = 0.0

    # --- Root sway (pelvis centroid XY RMS) -----------------------------
    if part_idx.get("pelvis"):
        pelvis_xy = _part_centroid(markers, part_idx["pelvis"])[:, :2]
        stats["root_sway_xy_rms"] = float(
            np.sqrt(((pelvis_xy - pelvis_xy.mean(axis=0)) ** 2).sum(axis=1).mean())
        )
    else:
        stats["root_sway_xy_rms"] = 0.0

    # --- Dropout / completeness -----------------------------------------
    if residual is not None and residual.size:
        dropped = float(np.mean(residual < 0))
        stats["marker_dropout_rate"] = dropped
    else:
        stats["marker_dropout_rate"] = 0.0

    # --- Duration --------------------------------------------------------
    stats["duration_sec"] = float(F / rate) if rate > 0 else 0.0
    stats["n_frames"] = float(F)
    return stats


def feature_names() -> list[str]:
    """Return the stable key order produced by :func:`compute_c3d_stats`
    on a canonical 57-marker Vicon Production clip.

    Used by the late-fusion model to allocate a fixed-size input vector.
    """
    # Build a synthetic dummy input and read keys back in insertion order.
    from diema.data.c3d_parser import VICON_PRODUCTION_MARKERS
    F, M = 12, len(VICON_PRODUCTION_MARKERS)
    dummy = np.zeros((F, M, 3), dtype=np.float32)
    stats = compute_c3d_stats(dummy, VICON_PRODUCTION_MARKERS, rate=120.0)
    return list(stats.keys())


def stats_to_vector(stats: dict[str, float], ordering: list[str] | None = None) -> np.ndarray:
    """Flatten a stats dict into a stable-ordered float32 vector.

    Missing keys fill with 0. Use :func:`feature_names` for the canonical ordering.
    """
    ordering = ordering or feature_names()
    return np.asarray([stats.get(k, 0.0) for k in ordering], dtype=np.float32)


__all__ = [
    "compute_c3d_stats",
    "feature_names",
    "stats_to_vector",
    "VICON_PRODUCTION_PART_GROUPS",
]
