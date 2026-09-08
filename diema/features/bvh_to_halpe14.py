"""DIEM-A BVH-24 → AIDE Halpe-14 2D projection for cross-dataset transfer.

Maps DIEM-A's 24-joint BVH quaternion skeleton (3D rotation per joint,
no face keypoints) to AIDE's 14-joint Halpe-style 2D image-plane keypoints
(x, y, confidence) at the resolution expected by UbH-GCN trained on AIDE.

Pipeline:
  1. Forward kinematics on BVH-24 → (F, 24, 3) world positions
  2. Orthographic project XY → (F, 24, 2) (drop Z-axis, camera looks down -Z)
  3. Joint subset map BVH-24 → UbH-GCN-14 body joints (10 mapped, 4 face joints zero-padded)
  4. Frame uniform-stride sample → 16 frames
  5. Per-channel statistics matching to AIDE training distribution
     (AIDE mean x=505 std 199, mean y=733 std 213, score in [0, 1])

Output:
  (3, T=16, V=14) float32, ready to feed `tmp/UbH-GCN-source/model/UbHGCN.Model`
  via the same (N, C=3, T, V=14, M=1) layout the AIDE feeder produces.

UbH-GCN-14 layout (0-indexed):
  0=Head 1=LEye 2=REye 3=LEar 4=REar
  5=RShoulder 6=LShoulder 7=RElbow 8=LElbow
  9=RWrist 10=LWrist 11=Nose(chin proxy)
  12=Neck 13=Hip-center

Face joints (1, 2, 3, 4, 11) are zero-padded with confidence=0 since
DIEM-A BVH has no face skeleton — AIDE's training distribution already
contained many low-confidence face detections, so confidence=0 is
distribution-consistent rather than out-of-distribution.

See: experiments/exp077_aide_feature_transfer/ (design rationale)
Related: diema/features/multi_stream.py (FK, parent_indices, offsets reused)
         tools/aide_to_skeleton_npy.py (UBH_GCN_TO_HALPE for AIDE side)
         tmp/UbH-GCN-source/model/UbHGCN.py (target consumer, has input data_bn)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from diema.features.multi_stream import compute_joint_positions

# UbH-GCN-14 (0-indexed target) → DIEM-A BVH-24 (0-indexed source), or
# None when the target joint has no BVH equivalent (face keypoints).
DIEMA_BVH_TO_UBH: dict[int, int | None] = {
    0:  7,    # Head      ← BVH Head           (pivot at top of neck1)
    1:  None, # LEye      ← face zero-pad (no BVH equivalent)
    2:  None, # REye      ← face zero-pad
    3:  None, # LEar      ← face zero-pad
    4:  None, # REar      ← face zero-pad
    5:  9,    # RShoulder ← BVH RightArm       (actual shoulder pivot, lateral from spine)
    6:  13,   # LShoulder ← BVH LeftArm
    7:  10,   # RElbow    ← BVH RightForeArm   (elbow pivot = end of upper arm)
    8:  14,   # LElbow    ← BVH LeftForeArm
    9:  11,   # RWrist    ← BVH RightHand      (wrist pivot = end of forearm)
    10: 15,   # LWrist    ← BVH LeftHand
    11: None, # Nose/chin ← face zero-pad
    12: 5,    # Neck      ← BVH Neck           (base-of-neck pivot, between shoulders)
    13: 0,    # Hip-center← BVH Hips
}

# AIDE training-set per-channel statistics (precomputed once over
# data/aide/processed/train_data.npy by inspecting the float32
# tensor). Used to scale DIEM-A projections so UbH-GCN's input BatchNorm
# sees inputs in a similar distribution to its training regime.
AIDE_TRAIN_STATS: dict[str, dict[str, float]] = {
    "x":     {"mean": 505.10, "std": 198.58},
    "y":     {"mean": 733.19, "std": 213.17},
    "score": {"mean": 0.25,   "std": 0.25},
}


def _uniform_stride_indices(num_frames: int, target_frames: int) -> np.ndarray:
    """Return target_frames indices spaced uniformly across [0, num_frames-1].

    Matches the spirit of AIDE's deterministic 16-from-45 sampling but
    generalises to arbitrary clip length.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")
    if num_frames == 1:
        return np.zeros(target_frames, dtype=np.int64)
    return np.round(np.linspace(0, num_frames - 1, target_frames)).astype(np.int64)


def _per_clip_normalize_to_aide(xy: np.ndarray) -> np.ndarray:
    """Map a per-clip (..., 2) array to AIDE's pixel-coordinate distribution.

    Per-clip: subtract clip mean (centering), divide by clip std (whitening),
    then multiply by AIDE std and add AIDE mean. **Caveat (a code review
    Risk 2)**: per-clip whitening forces every clip — small contempt
    gestures and big anger swings alike — to identical pixel-space variance,
    destroying cross-clip motion-energy variance that is informative for
    emotion classification. See `_dataset_affine_to_aide` for an
    alternative that preserves the variance.
    """
    if xy.shape[-1] != 2:
        raise ValueError(f"expected last dim 2, got shape {xy.shape}")
    x = xy[..., 0]
    y = xy[..., 1]
    eps = 1e-6
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.std() + eps
    sigma_y = y.std() + eps
    x_aide = (x - mu_x) / sigma_x * AIDE_TRAIN_STATS["x"]["std"] + AIDE_TRAIN_STATS["x"]["mean"]
    y_aide = (y - mu_y) / sigma_y * AIDE_TRAIN_STATS["y"]["std"] + AIDE_TRAIN_STATS["y"]["mean"]
    return np.stack([x_aide, y_aide], axis=-1).astype(np.float32)


def _dataset_affine_to_aide(
    xy: np.ndarray,
    diema_stats: dict[str, dict[str, float]] | None = None,
) -> np.ndarray:
    """Map (..., 2) to AIDE distribution via a single dataset-level affine.

    Preserves cross-clip variance, unlike `_per_clip_normalize_to_aide`.
    The affine is `(x - μ_diema) / σ_diema * σ_aide + μ_aide`, using
    *dataset-level* μ/σ measured once on all DIEM-A clips.
    """
    if xy.shape[-1] != 2:
        raise ValueError(f"expected last dim 2, got shape {xy.shape}")
    if diema_stats is None:
        diema_stats = DIEMA_GLOBAL_STATS
    x = xy[..., 0]
    y = xy[..., 1]
    eps = 1e-6
    x_aide = (x - diema_stats["x"]["mean"]) / (diema_stats["x"]["std"] + eps) * AIDE_TRAIN_STATS["x"]["std"] + AIDE_TRAIN_STATS["x"]["mean"]
    y_aide = (y - diema_stats["y"]["mean"]) / (diema_stats["y"]["std"] + eps) * AIDE_TRAIN_STATS["y"]["std"] + AIDE_TRAIN_STATS["y"]["mean"]
    return np.stack([x_aide, y_aide], axis=-1).astype(np.float32)


# Dataset-level statistics of DIEM-A FK-projected joint coordinates,
# precomputed once over all 7992 clips after `drop_axis='y',
# flip_vertical=True`. Used by `normalize="dataset_affine"`. Values are
# placeholder zeros; populated by calling `compute_diema_global_stats()`.
DIEMA_GLOBAL_STATS: dict[str, dict[str, float]] = {
    "x": {"mean": -0.5614, "std": 11.9717},   # precomputed via compute_diema_global_stats over 7992 clips
    "y": {"mean": -12.8517, "std": 11.0322},  # drop_axis="y", flip_vertical=True body joints only
}


def compute_diema_global_stats(
    motion_npz_path: str,
    drop_axis: str = "y",
    flip_vertical: bool = True,
) -> dict[str, dict[str, float]]:
    """Precompute DIEM-A dataset-level x/y mean/std over all body joints.

    Run once and use the returned stats to seed DIEMA_GLOBAL_STATS.
    """
    npz = np.load(motion_npz_path, allow_pickle=True)
    n = int(npz["num_clips"])
    all_xs = []
    all_ys = []
    for i in range(n):
        rp = np.asarray(npz[f"clip_{i}_root_pos"], dtype=np.float32)
        jq = np.asarray(npz[f"clip_{i}_joint_data"], dtype=np.float32)
        # Recompute FK (skip per-clip whitening — just raw projected coords)
        world_pos = compute_joint_positions(rp, jq)
        if drop_axis == "y":
            xy_3d = world_pos[:, :, [0, 2]].copy()
        elif drop_axis == "z":
            xy_3d = world_pos[:, :, [0, 1]].copy()
        elif drop_axis == "x":
            xy_3d = world_pos[:, :, [1, 2]].copy()
        else:
            raise ValueError(drop_axis)
        if flip_vertical:
            xy_3d[:, :, 1] = -xy_3d[:, :, 1]
        # Only consider body joints (avoid face-padded zeros)
        body_bvh = [v for v in DIEMA_BVH_TO_UBH.values() if v is not None]
        xy_body = xy_3d[:, body_bvh, :]  # (F, n_body, 2)
        all_xs.append(xy_body[..., 0].ravel())
        all_ys.append(xy_body[..., 1].ravel())
    x_all = np.concatenate(all_xs)
    y_all = np.concatenate(all_ys)
    return {
        "x": {"mean": float(x_all.mean()), "std": float(x_all.std())},
        "y": {"mean": float(y_all.mean()), "std": float(y_all.std())},
    }


def project_bvh_to_halpe14(
    root_pos: np.ndarray,
    joint_quats: np.ndarray,
    target_frames: int = 16,
    skeleton_info: dict[str, Any] | None = None,
    drop_axis: str = "y",
    body_confidence: float = 1.0,
    face_confidence: float = 0.0,
    face_coord_source: str = "zero",
    flip_vertical: bool = True,
    normalize: str = "per_clip",
) -> np.ndarray:
    """Project a DIEM-A clip's BVH skeleton to AIDE Halpe-14 2D layout.

    Args:
        root_pos: (F, 3) per-frame root positions (Hips world coords).
        joint_quats: (F, 24, 4) per-frame quaternions (wxyz).
        target_frames: number of output frames T (default 16, matches AIDE).
        skeleton_info: optional dict with 'offsets' / 'parent_indices'.
        drop_axis: which BVH axis to drop for orthographic projection. "z"
            drops Z (camera looking down -Z = front-facing view), "y" drops
            Y (top-down view).
        body_confidence: confidence value for joints that have a BVH source.
        face_confidence: confidence value for face joints (low/zero).
        flip_vertical: when True (default), negates the BVH "up" axis after
            projection so the resulting y-channel follows image convention
            (screen-y increases downward; head has lower y than hip).
            Without this flip, BVH Z-up gives head-high / hip-low which is
            inverted relative to AIDE's image-plane Halpe coords.
        face_coord_source: how to populate face joint coordinates. "zero"
            (DEFAULT, empirically best) leaves them at (0, 0) — this matches
            AIDE's training distribution where face joints often had
            low-confidence near-zero detections. "head" copies BVH Head
            coords (proposed by a review finding to avoid a bimodal
            x distribution at data_bn) — empirically REJECTED: caused
            uniform −1.7 to −0.4pp degradation across 9 probe variants
            because it forces face joints to mimic head motion (information
            redundancy + bone-stream face-to-face bones become trivially
            zero). See `results_variants_b01.json` for the comparison.

    Returns:
        (3, T, 14) float32 tensor:
          channel 0 = x pixel coord (matched to AIDE x distribution)
          channel 1 = y pixel coord (matched to AIDE y distribution)
          channel 2 = confidence
    """
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must be (F, 3), got {root_pos.shape}")
    if joint_quats.ndim != 3 or joint_quats.shape[1:] != (24, 4):
        raise ValueError(f"joint_quats must be (F, 24, 4), got {joint_quats.shape}")
    if root_pos.shape[0] != joint_quats.shape[0]:
        raise ValueError(
            f"root_pos F={root_pos.shape[0]} != joint_quats F={joint_quats.shape[0]}"
        )

    world_pos = compute_joint_positions(
        root_pos,
        joint_quats,
        offsets=None if skeleton_info is None else skeleton_info.get("offsets"),
        parents=None if skeleton_info is None else skeleton_info.get("parent_indices"),
    )  # (F, 24, 3)

    if drop_axis == "z":
        # Keep (X=lateral, Y=depth) — bird's-eye / top-down. NOT the
        # AIDE camera view; documented for ablation only.
        xy_3d = world_pos[:, :, [0, 1]].copy()
    elif drop_axis == "y":
        # Keep (X=lateral, Z=vertical) — true face-front projection
        # matching AIDE's camera viewpoint (BVH is Z-up Maya/MotionBuilder
        # convention; Hips→Spine offset is purely +Z). This is the
        # recommended setting.
        xy_3d = world_pos[:, :, [0, 2]].copy()
    elif drop_axis == "x":
        xy_3d = world_pos[:, :, [1, 2]].copy()
    else:
        raise ValueError(f"drop_axis must be x|y|z, got {drop_axis!r}")
    if flip_vertical:
        # Negate the second channel (the "y" of the 2D output) so screen-y
        # follows image convention (increases downward). For drop_axis="y"
        # this flips BVH-Z so head goes to small y (top of frame) and hips
        # go to large y (bottom of frame), matching AIDE's pixel layout.
        xy_3d[:, :, 1] = -xy_3d[:, :, 1]

    F = xy_3d.shape[0]
    frame_idx = _uniform_stride_indices(F, target_frames)  # (T,)
    xy_sampled = xy_3d[frame_idx]  # (T, 24, 2)

    out = np.zeros((3, target_frames, 14), dtype=np.float32)
    body_mask = np.zeros(14, dtype=bool)

    for ubh_idx in range(14):
        bvh_idx = DIEMA_BVH_TO_UBH[ubh_idx]
        if bvh_idx is None:
            continue
        out[0:2, :, ubh_idx] = xy_sampled[:, bvh_idx, :].T  # (2, T)
        body_mask[ubh_idx] = True

    body_indices = np.flatnonzero(body_mask)
    out_body_2d = np.transpose(out[0:2][:, :, body_indices], (1, 2, 0))  # (T, n_body, 2)
    if normalize == "per_clip":
        out_body_2d_normed_view = _per_clip_normalize_to_aide(out_body_2d)
    elif normalize == "dataset_affine":
        out_body_2d_normed_view = _dataset_affine_to_aide(out_body_2d)
    elif normalize == "none":
        out_body_2d_normed_view = out_body_2d.astype(np.float32)
    else:
        raise ValueError(f"normalize must be per_clip|dataset_affine|none, got {normalize!r}")
    out_body_2d_normed = np.transpose(out_body_2d_normed_view, (2, 0, 1))
    for j, idx in enumerate(body_indices):
        out[0:2, :, idx] = out_body_2d_normed[:, :, j]

    out[2, :, body_indices] = body_confidence
    face_indices = np.flatnonzero(~body_mask)
    out[2, :, face_indices] = face_confidence

    if face_coord_source == "head":
        # Copy normalized BVH Head coords (UbH idx 0) into face slots so
        # data_bn sees a unimodal distribution; confidence=0 signals
        # unreliability.
        head_x = out[0, :, 0]  # (T,) — already AIDE-normalized
        head_y = out[1, :, 0]
        for idx in face_indices:
            out[0, :, idx] = head_x
            out[1, :, idx] = head_y
    elif face_coord_source == "zero":
        pass  # leave at 0,0 — original behavior, pre-Opus-4.6-review
    else:
        raise ValueError(f"face_coord_source must be 'head' or 'zero', got {face_coord_source!r}")

    return out


def project_batch(
    root_pos_list: list[np.ndarray],
    joint_quats_list: list[np.ndarray],
    target_frames: int = 16,
    **kwargs,
) -> np.ndarray:
    """Vectorised wrapper: project a list of clips → (N, 3, T, 14, M=1)."""
    if len(root_pos_list) != len(joint_quats_list):
        raise ValueError("root_pos_list and joint_quats_list must have same length")
    N = len(root_pos_list)
    out = np.zeros((N, 3, target_frames, 14, 1), dtype=np.float32)
    for i, (rp, jq) in enumerate(zip(root_pos_list, joint_quats_list)):
        out[i, :, :, :, 0] = project_bvh_to_halpe14(rp, jq, target_frames=target_frames, **kwargs)
    return out
