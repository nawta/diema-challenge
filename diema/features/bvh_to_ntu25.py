"""DIEM-A BVH-24 → MAMP NTU-RGB+D 25-joint 3D projection for cross-dataset transfer.

Maps DIEM-A's 24-joint BVH quaternion skeleton (3D rotation per joint) to
the NTU-RGB+D 25-joint 3D layout that MAMP (ICCV 2023, masked motion
prediction) was pretrained on.

NTU-RGB+D 25-joint order (0-indexed; NTU docs are 1-indexed):
    0  SpineBase        9  RightElbow      18 RightAnkle
    1  SpineMid        10  RightWrist      19 RightFoot
    2  Neck            11  RightHand       20 SpineShoulder
    3  Head            12  LeftHip         21 LeftHandTip
    4  LeftShoulder    13  LeftKnee        22 LeftThumb
    5  LeftElbow       14  LeftAnkle       23 RightHandTip
    6  LeftWrist       15  LeftFoot        24 RightThumb
    7  LeftHand        16  RightHip
    8  RightShoulder   17  RightKnee

BVH-24 chain semantics: a BVH joint's position is its origin, so the
"shoulder" rotation joint (BVH *Arm) is the NTU shoulder, BVH *ForeArm is
the NTU elbow, BVH *Hand is the NTU wrist. DIEM-A BVH-24 has no finger
detail, so NTU hand/handtip/thumb joints (7,11,21,22,23,24) duplicate the
corresponding wrist/hand joint — the standard approach when the source
skeleton is coarser than the pretrained model's layout.

Pipeline:
  1. Forward kinematics on BVH-24 → (F, 24, 3) world positions
     (reuses diema.features.multi_stream.compute_joint_positions, same as
      bvh_to_h36m17.py)
  2. Subset/duplicate BVH-24 → NTU-25
  3. Center on NTU SpineMid (idx 1) per frame — matches the CTR-GCN /
     pyskl pre_normalization convention MAMP's NTU data follows
  4. Scale: "unit" (per-clip scale to [-1,1], analogous to exp081's
     validated MotionBERT convention) or "none" (raw FK units)
  5. Resample F → target_frames (default 120 = MAMP num_frames)

Output: (T, 25, 3) float32 — 3D (x, y, z).

See: experiments/exp085_mamp_frozen_probe/ , tmp/MAMP/feeder/feeder_ntu.py
Related: diema/features/bvh_to_h36m17.py (parallel MotionBERT projector)
         tmp/MAMP/model/transformer.py (downstream encoder; (N,C,T,V,M) input)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from diema.features.multi_stream import compute_joint_positions

# NTU-25 (0-indexed) ← BVH-24 source index.
# Hand tips / thumbs duplicate the wrist/hand (BVH-24 has no fingers).
NTU25_TO_BVH: dict[int, int] = {
    0: 0,    # SpineBase     ← Hips
    1: 2,    # SpineMid      ← Spine1
    2: 5,    # Neck          ← Neck
    3: 7,    # Head          ← Head
    4: 13,   # LeftShoulder  ← LeftArm   (shoulder rotation joint)
    5: 14,   # LeftElbow     ← LeftForeArm
    6: 15,   # LeftWrist     ← LeftHand
    7: 15,   # LeftHand      ← LeftHand  (dup)
    8: 9,    # RightShoulder ← RightArm
    9: 10,   # RightElbow    ← RightForeArm
    10: 11,  # RightWrist    ← RightHand
    11: 11,  # RightHand     ← RightHand (dup)
    12: 20,  # LeftHip       ← LeftUpLeg
    13: 21,  # LeftKnee      ← LeftLeg
    14: 22,  # LeftAnkle     ← LeftFoot
    15: 23,  # LeftFoot      ← LeftToeBase
    16: 16,  # RightHip      ← RightUpLeg
    17: 17,  # RightKnee     ← RightLeg
    18: 18,  # RightAnkle    ← RightFoot
    19: 19,  # RightFoot     ← RightToeBase
    20: 4,   # SpineShoulder ← Spine3
    21: 15,  # LeftHandTip   ← LeftHand  (dup)
    22: 15,  # LeftThumb     ← LeftHand  (dup)
    23: 11,  # RightHandTip  ← RightHand (dup)
    24: 11,  # RightThumb    ← RightHand (dup)
}

_NTU_CENTER_JOINT = 1  # NTU SpineMid (0-indexed) — CTR-GCN pre_normalization center


def _uniform_stride_indices(num_frames: int, target_frames: int) -> np.ndarray:
    if num_frames <= 0 or target_frames <= 0:
        raise ValueError(f"num_frames {num_frames}, target_frames {target_frames}")
    if num_frames == 1:
        return np.zeros(target_frames, dtype=np.int64)
    return np.round(np.linspace(0, num_frames - 1, target_frames)).astype(np.int64)


def project_bvh_to_ntu25(
    root_pos: np.ndarray,
    joint_quats: np.ndarray,
    target_frames: int = 120,
    skeleton_info: dict[str, Any] | None = None,
    normalize: str = "unit",
) -> np.ndarray:
    """Project a DIEM-A clip's BVH skeleton to MAMP NTU-25 3D layout.

    Args:
        root_pos: (F, 3) per-frame Hips world positions.
        joint_quats: (F, 24, 4) per-frame quaternions (wxyz).
        target_frames: output frame count T (default 120 = MAMP num_frames).
        skeleton_info: optional dict with 'offsets' / 'parent_indices'.
        normalize: "unit" → center on SpineMid + per-clip scale to [-1,1]
            (analogous to exp081 MotionBERT convention; default);
            "center" → center on SpineMid only (raw FK scale);
            "none" → raw FK world positions.

    Returns:
        (T, 25, 3) float32.
    """
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must be (F, 3), got {root_pos.shape}")
    if joint_quats.ndim != 3 or joint_quats.shape[1:] != (24, 4):
        raise ValueError(f"joint_quats must be (F, 24, 4), got {joint_quats.shape}")
    if root_pos.shape[0] != joint_quats.shape[0]:
        raise ValueError(
            f"root_pos F={root_pos.shape[0]} != joint_quats F={joint_quats.shape[0]}"
        )
    if normalize not in ("unit", "center", "none"):
        raise ValueError(f"normalize must be unit|center|none, got {normalize!r}")

    world_pos = compute_joint_positions(
        root_pos,
        joint_quats,
        offsets=None if skeleton_info is None else skeleton_info.get("offsets"),
        parents=None if skeleton_info is None else skeleton_info.get("parent_indices"),
    )  # (F, 24, 3)

    F = world_pos.shape[0]
    frame_idx = _uniform_stride_indices(F, target_frames)
    sampled = world_pos[frame_idx]  # (T, 24, 3)

    T = target_frames
    out = np.zeros((T, 25, 3), dtype=np.float32)
    for ntu_idx, bvh_idx in NTU25_TO_BVH.items():
        out[:, ntu_idx, :] = sampled[:, bvh_idx, :]

    if normalize in ("unit", "center"):
        center = out[:, _NTU_CENTER_JOINT : _NTU_CENTER_JOINT + 1, :].copy()
        out = out - center
    if normalize == "unit":
        scale = np.maximum(np.abs(out).max(), 1e-6)
        out = out / scale

    return out.astype(np.float32)


def project_batch(
    root_pos_list: list[np.ndarray],
    joint_quats_list: list[np.ndarray],
    target_frames: int = 120,
    **kwargs,
) -> np.ndarray:
    """Vectorised wrapper: list of clips → (N, T, 25, 3)."""
    if len(root_pos_list) != len(joint_quats_list):
        raise ValueError("root_pos_list and joint_quats_list must have same length")
    N = len(root_pos_list)
    out = np.zeros((N, target_frames, 25, 3), dtype=np.float32)
    for i, (rp, jq) in enumerate(zip(root_pos_list, joint_quats_list)):
        out[i] = project_bvh_to_ntu25(rp, jq, target_frames=target_frames, **kwargs)
    return out
