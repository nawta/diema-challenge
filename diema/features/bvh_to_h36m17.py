"""DIEM-A BVH-24 → MotionBERT H36M-17 2D projection for cross-dataset transfer.

Maps DIEM-A's 24-joint BVH quaternion skeleton (3D rotation per joint) to
MotionBERT's 17-joint H36M layout (2D + confidence, normalized to [-1, 1]).

H36M-17 joint order (per MotionBERT lib/data/dataset_action.py:coco2h36m):
    0: root  (= midpoint of L/R hip)
    1: rhip
    2: rkne
    3: rank
    4: lhip
    5: lkne
    6: lank
    7: belly (= midpoint of root + neck)
    8: neck  (= midpoint of L/R shoulder)
    9: nose
    10: head (= midpoint of L/R eye)
    11: lsho
    12: lelb
    13: lwri
    14: rsho
    15: relb
    16: rwri

Pipeline:
  1. Forward kinematics on BVH-24 → (F, 24, 3) world positions
  2. Orthographic project XZ (drop_axis="y") with flip_vertical=True
     (matches exp077 face-front convention validated for MotionBERT-style
     image-plane data)
  3. Subset BVH-24 → H36M-17 (synthesize 'root', 'belly', 'neck' midpoints;
     no 'nose' equivalent so synthesize from head + small offset, with
     confidence=0.5 to mark uncertainty)
  4. Root-relative: subtract root joint (H36M idx 0) per frame
  5. Scale to [-1, 1] per clip (matches MotionBERT 'rootrel + scale [-1,1]'
     training convention)
  6. Concat confidence channel (1.0 for confident, 0.5 for nose-synthesis)

Output: (T, 17, 3) float32 — channels are (x_normalized, y_normalized, conf).

See: experiments/exp081_motionbert_transfer
Related: diema/features/bvh_to_halpe14.py (parallel projector for AIDE)
         tmp/MotionBERT/lib/data/dataset_action.py (H36M layout reference)
         tmp/MotionBERT/configs/pretrain/MB_lite.yaml (rootrel: True, num_joints: 17)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from diema.features.multi_stream import compute_joint_positions


# DIEM-A BVH-24 joint indices (per diema/features/multi_stream.py):
# 0 Hips, 1 Spine, 2 Spine1, 3 Spine2, 4 Spine3,
# 5 Neck, 6 Neck1, 7 Head,
# 8 RightShoulder, 9 RightArm, 10 RightForeArm, 11 RightHand,
# 12 LeftShoulder, 13 LeftArm, 14 LeftForeArm, 15 LeftHand,
# 16 RightUpLeg, 17 RightLeg, 18 RightFoot, 19 RightToeBase,
# 20 LeftUpLeg, 21 LeftLeg, 22 LeftFoot, 23 LeftToeBase
#
# H36M target idx → (bvh source idx | None for synthesized | tuple to average)
# None means "synthesize from other joints" (handled per-key below).
H36M_TO_BVH: dict[int, int | tuple | str] = {
    0:  "root_synth",     # midpoint of (16 RightUpLeg, 20 LeftUpLeg)
    1:  16,               # rhip ← RightUpLeg pivot
    2:  17,               # rkne ← RightLeg pivot
    3:  18,               # rank ← RightFoot pivot
    4:  20,               # lhip ← LeftUpLeg pivot
    5:  21,               # lkne ← LeftLeg pivot
    6:  22,               # lank ← LeftFoot pivot
    7:  "belly_synth",    # belly ← midpoint of (root, neck)
    8:  5,                # neck ← BVH Neck (base of neck) — synthesizes well as shoulder-center
    9:  "nose_synth",     # nose ← head + small offset (no BVH equiv; conf=0.5)
    10: 7,                # head ← BVH Head pivot
    11: 13,               # lsho ← LeftArm pivot (actual shoulder)
    12: 14,               # lelb ← LeftForeArm pivot
    13: 15,               # lwri ← LeftHand pivot
    14: 9,                # rsho ← RightArm pivot
    15: 10,               # relb ← RightForeArm pivot
    16: 11,               # rwri ← RightHand pivot
}


def _uniform_stride_indices(num_frames: int, target_frames: int) -> np.ndarray:
    if num_frames <= 0 or target_frames <= 0:
        raise ValueError(f"num_frames {num_frames}, target_frames {target_frames}")
    if num_frames == 1:
        return np.zeros(target_frames, dtype=np.int64)
    return np.round(np.linspace(0, num_frames - 1, target_frames)).astype(np.int64)


def project_bvh_to_h36m17(
    root_pos: np.ndarray,
    joint_quats: np.ndarray,
    target_frames: int = 243,
    skeleton_info: dict[str, Any] | None = None,
    drop_axis: str = "y",
    flip_vertical: bool = True,
    scale_to_unit: bool = True,
    synth_nose_offset_ratio: float = 0.05,
) -> np.ndarray:
    """Project a DIEM-A clip's BVH skeleton to MotionBERT H36M-17 layout.

    Args:
        root_pos: (F, 3) per-frame Hips world positions.
        joint_quats: (F, 24, 4) per-frame quaternions (wxyz).
        target_frames: number of output frames T (default 243 = MB maxlen).
        skeleton_info: optional dict with 'offsets' / 'parent_indices'.
        drop_axis: which BVH axis to drop (default "y" = depth, keep X=lateral, Z=vertical).
        flip_vertical: when True, negates kept "vertical" axis so screen-y
            increases downward (image convention).
        scale_to_unit: when True, rescales coords to [-1, 1] per clip
            (matches MotionBERT scale_range=[1,1] convention).
        synth_nose_offset_ratio: nose joint is synthesized as head + small
            upward offset in normalized units (no BVH equivalent; conf=0.5).

    Returns:
        (T, 17, 3) float32 — channels (x, y, confidence).
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

    if drop_axis == "y":
        xy_3d = world_pos[:, :, [0, 2]].copy()
    elif drop_axis == "z":
        xy_3d = world_pos[:, :, [0, 1]].copy()
    elif drop_axis == "x":
        xy_3d = world_pos[:, :, [1, 2]].copy()
    else:
        raise ValueError(f"drop_axis must be x|y|z, got {drop_axis!r}")
    if flip_vertical:
        xy_3d[:, :, 1] = -xy_3d[:, :, 1]

    F = xy_3d.shape[0]
    frame_idx = _uniform_stride_indices(F, target_frames)
    xy_sampled = xy_3d[frame_idx]  # (T, 24, 2)

    T = target_frames
    out_xy = np.zeros((T, 17, 2), dtype=np.float32)
    out_conf = np.ones((T, 17), dtype=np.float32)
    for h36m_idx, src in H36M_TO_BVH.items():
        if isinstance(src, int):
            out_xy[:, h36m_idx, :] = xy_sampled[:, src, :]
        elif src == "root_synth":
            out_xy[:, h36m_idx, :] = 0.5 * (xy_sampled[:, 16, :] + xy_sampled[:, 20, :])
        elif src == "belly_synth":
            pass
        elif src == "nose_synth":
            out_xy[:, h36m_idx, :] = xy_sampled[:, 7, :]
            out_conf[:, h36m_idx] = 0.5
        else:
            raise RuntimeError(f"unknown H36M_TO_BVH source: {src!r}")

    out_xy[:, 7, :] = 0.5 * (out_xy[:, 0, :] + out_xy[:, 8, :])

    out_xy = out_xy - out_xy[:, 0:1, :]

    if scale_to_unit:
        scale = np.maximum(np.abs(out_xy).max(), 1e-6)
        out_xy = out_xy / scale

    if H36M_TO_BVH[9] == "nose_synth":
        out_xy[:, 9, 1] = out_xy[:, 10, 1] + synth_nose_offset_ratio
        out_xy[:, 9, 0] = out_xy[:, 10, 0]

    out = np.concatenate([out_xy, out_conf[..., None]], axis=-1).astype(np.float32)
    return out


def project_batch(
    root_pos_list: list[np.ndarray],
    joint_quats_list: list[np.ndarray],
    target_frames: int = 243,
    **kwargs,
) -> np.ndarray:
    """Vectorised wrapper: list of clips → (N, T, 17, 3)."""
    if len(root_pos_list) != len(joint_quats_list):
        raise ValueError("root_pos_list and joint_quats_list must have same length")
    N = len(root_pos_list)
    out = np.zeros((N, target_frames, 17, 3), dtype=np.float32)
    for i, (rp, jq) in enumerate(zip(root_pos_list, joint_quats_list)):
        out[i] = project_bvh_to_h36m17(rp, jq, target_frames=target_frames, **kwargs)
    return out
