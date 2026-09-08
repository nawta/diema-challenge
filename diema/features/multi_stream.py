"""Multi-stream feature pipeline for skeleton-based emotion recognition.

Computes several input representations from quaternion+root_pos data:
  - rotation_6d: 6D continuous rotation per joint (current default, C=6, V=25)
  - joint_pos:   root-relative 3D positions (FK-based, C=3, V=25)
  - bone:        child - parent positional vectors (C=3, V=25)
  - joint_motion: frame-to-frame differences of joint_pos (C=3, V=25)
  - bone_motion:  frame-to-frame differences of bone (C=3, V=25)

All streams use the **25-node CTV layout**:
  node 0 = virtual root (root_pos / 0 for non-position streams)
  nodes 1-24 = BVH joints in pybvh order (Hips, Spine, ..., LeftToeBase)

Related: diema/data/dataset.py (MotionDataset.stream_type)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Skeleton offsets (cached from a reference BVH file)
# ---------------------------------------------------------------------------

# Reference BVH offsets for the DIEM-A skeleton (24 BVH joints, in pybvh order).
# Obtained once from data/diema_challenge/raw/bvh/train/JP_06_anger_1_H.bvh.
# These are the rest-pose local offsets used for forward kinematics.
# Shape: (24, 3)
_DIEMA_OFFSETS_BVH: np.ndarray | None = None


def _load_reference_offsets() -> np.ndarray:
    """Load the DIEM-A skeleton offsets (24 joints) from a reference BVH.

    Cached after first call. Assumes all DIEM-A samples share the same
    skeleton structure, which we verified via EDA.
    """
    global _DIEMA_OFFSETS_BVH
    if _DIEMA_OFFSETS_BVH is not None:
        return _DIEMA_OFFSETS_BVH

    # Hard-coded values from the reference BVH (extracted via pybvh).
    # This avoids depending on pybvh / BVH files at runtime.
    offsets = np.array([
        [0.0,      0.0,      0.0],      # 0: Hips
        [0.0,      0.0,      4.45284],  # 1: Spine
        [0.0,     -0.77323,  4.38519],  # 2: Spine1
        [0.0,     -0.38809,  4.43589],  # 3: Spine2
        [0.0,      0.0,      4.45284],  # 4: Spine3
        [0.0,      0.57251,  5.72508],  # 5: Neck
        [0.0,      0.14369,  1.6424],   # 6: Neck1
        [0.0,      0.28629,  1.62363],  # 7: Head
        [0.0,      0.0,      4.00755],  # 8: RightShoulder
        [7.25812,  0.0,      0.0],      # 9: RightArm
        [11.28126, 0.0,      0.0],      # 10: RightForeArm
        [8.65628,  0.0,      0.0],      # 11: RightHand
        [0.0,      0.0,      4.00755],  # 12: LeftShoulder
        [-7.25812, 0.0,      0.0],      # 13: LeftArm
        [-11.28126, 0.0,     0.0],      # 14: LeftForeArm
        [-8.65628, 0.0,      0.0],      # 15: LeftHand
        [3.94455,  0.0,      0.0],      # 16: RightUpLeg
        [0.0,      0.0,     -16.5979],  # 17: RightLeg
        [0.0,      0.0,     -15.90531], # 18: RightFoot
        [0.0,      4.84054, -1.23599],  # 19: RightToeBase
        [-3.94455, 0.0,      0.0],      # 20: LeftUpLeg
        [0.0,      0.0,     -16.5979],  # 21: LeftLeg
        [0.0,      0.0,     -15.90531], # 22: LeftFoot
        [0.0,      4.84054, -1.23599],  # 23: LeftToeBase
    ], dtype=np.float32)
    _DIEMA_OFFSETS_BVH = offsets
    return offsets


# Parent indices for BVH 24-joint skeleton (-1 for root).
# Derived from the BVH hierarchy. Index i's parent is _DIEMA_PARENTS_BVH[i].
_DIEMA_PARENTS_BVH: list[int] = [
    -1,   # 0: Hips
    0,    # 1: Spine        → Hips
    1,    # 2: Spine1       → Spine
    2,    # 3: Spine2       → Spine1
    3,    # 4: Spine3       → Spine2
    4,    # 5: Neck         → Spine3
    5,    # 6: Neck1        → Neck
    6,    # 7: Head         → Neck1
    4,    # 8: RightShoulder → Spine3
    8,    # 9: RightArm
    9,    # 10: RightForeArm
    10,   # 11: RightHand
    4,    # 12: LeftShoulder → Spine3
    12,   # 13: LeftArm
    13,   # 14: LeftForeArm
    14,   # 15: LeftHand
    0,    # 16: RightUpLeg   → Hips
    16,   # 17: RightLeg
    17,   # 18: RightFoot
    18,   # 19: RightToeBase
    0,    # 20: LeftUpLeg    → Hips
    20,   # 21: LeftLeg
    21,   # 22: LeftFoot
    22,   # 23: LeftToeBase
]


# ---------------------------------------------------------------------------
# Quaternion helpers (numpy, vectorized over frames)
# ---------------------------------------------------------------------------

def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two (F, 4) quaternion arrays. Both (w, x, y, z)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def _quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3D vectors by quaternions. q: (F, 4), v: (3,) or (F, 3). Returns (F, 3)."""
    qw = q[..., 0:1]
    qxyz = q[..., 1:]
    # v_rot = v + 2*cross(qxyz, cross(qxyz, v) + qw*v)
    v_arr = np.broadcast_to(np.asarray(v, dtype=q.dtype), q[..., :3].shape)
    t = 2.0 * np.cross(qxyz, v_arr)
    return v_arr + qw * t + np.cross(qxyz, t)


# ---------------------------------------------------------------------------
# Forward kinematics and stream computation
# ---------------------------------------------------------------------------

def compute_joint_positions(
    root_pos: np.ndarray,
    joint_quats: np.ndarray,
    offsets: np.ndarray | None = None,
    parents: list[int] | None = None,
) -> np.ndarray:
    """Forward kinematics: compute world-frame 3D positions per joint.

    Args:
        root_pos: (F, 3) per-frame root position
        joint_quats: (F, J, 4) per-frame per-joint quaternions (wxyz)
        offsets: (J, 3) rest-pose local offsets. Defaults to DIEMA 24-joint values.
        parents: parent indices list (-1 for root). Defaults to DIEMA 24 parents.

    Returns:
        (F, J, 3) world-frame joint positions (Hips == root_pos).
    """
    if offsets is None:
        offsets = _load_reference_offsets()
    if parents is None:
        parents = _DIEMA_PARENTS_BVH

    F, J, _ = joint_quats.shape
    world_rot = np.zeros((F, J, 4), dtype=np.float32)
    world_pos = np.zeros((F, J, 3), dtype=np.float32)

    for i in range(J):
        p = parents[i]
        if p == -1:
            world_rot[:, i] = joint_quats[:, i]
            world_pos[:, i] = root_pos
        else:
            # Position: parent_pos + rotate offset by parent's world rotation
            off = offsets[i].astype(np.float32)
            rotated = _quat_rotate_vector(world_rot[:, p], off)
            world_pos[:, i] = world_pos[:, p] + rotated
            # Rotation: compose parent's world rotation with local quaternion
            world_rot[:, i] = _quat_multiply(world_rot[:, p], joint_quats[:, i])

    return world_pos


def compute_bone_vectors(
    joint_positions: np.ndarray,
    parents: list[int] | None = None,
) -> np.ndarray:
    """Bone vectors as child position minus parent position.

    Args:
        joint_positions: (F, J, 3)
        parents: parent indices (-1 for root). Root bone is zero vector.

    Returns:
        (F, J, 3) bone vectors.
    """
    if parents is None:
        parents = _DIEMA_PARENTS_BVH
    F, J, _ = joint_positions.shape
    bones = np.zeros_like(joint_positions)
    for i, p in enumerate(parents):
        if p == -1:
            bones[:, i] = 0.0
        else:
            bones[:, i] = joint_positions[:, i] - joint_positions[:, p]
    return bones


def compute_temporal_differences(features: np.ndarray) -> np.ndarray:
    """Frame-to-frame differences. First frame is zero-padded so the output
    has the same shape as input.

    Args:
        features: (F, ...) array, differences are taken along axis 0.

    Returns:
        Same shape as input.
    """
    diffs = np.zeros_like(features)
    diffs[1:] = features[1:] - features[:-1]
    return diffs


# ---------------------------------------------------------------------------
# 25-node CTV packing
# ---------------------------------------------------------------------------

def _pack_25_node(
    root_pos: np.ndarray,
    joint_feats: np.ndarray,
) -> np.ndarray:
    """Pack (root_pos, joint_feats) into CTV layout (C, T, 25).

    Node 0 stores root_pos zero-padded to match channel count,
    nodes 1-24 store joint_feats in BVH order.

    Args:
        root_pos: (F, 3)
        joint_feats: (F, 24, C) where C is the feature channel count

    Returns:
        (C, F, 25) packed tensor
    """
    F, J, C = joint_feats.shape
    assert J == 24, f"expected 24 BVH joints, got {J}"

    # Node 0: root_pos (optionally padded if C > 3)
    if C == 3:
        node0 = root_pos.astype(np.float32)  # (F, 3)
    elif C > 3:
        node0 = np.zeros((F, C), dtype=np.float32)
        node0[:, :3] = root_pos
    else:
        node0 = np.zeros((F, C), dtype=np.float32)
    node0 = node0[:, None, :]  # (F, 1, C)

    combined = np.concatenate([node0, joint_feats], axis=1)  # (F, 25, C)
    return combined.transpose(2, 0, 1).astype(np.float32)    # (C, F, 25)


def get_stream_features(
    root_pos: np.ndarray,
    joint_quats: np.ndarray,
    skeleton_info: dict[str, Any] | None = None,
    stream_type: str = "joint_pos",
) -> tuple[np.ndarray, int]:
    """Compute a specific feature stream and pack to (C, F, 25) CTV layout.

    Args:
        root_pos: (F, 3) per-frame root positions
        joint_quats: (F, 24, 4) per-frame quaternions (wxyz)
        skeleton_info: dict with optional keys 'offsets', 'parent_indices'.
            Falls back to DIEMA reference values if absent.
        stream_type: one of 'joint_pos', 'bone', 'joint_motion', 'bone_motion'

    Returns:
        (data_ctv, in_channels) where data_ctv has shape (C, F, 25).
    """
    offsets = None
    parents = None
    if skeleton_info is not None:
        if "offsets" in skeleton_info and skeleton_info["offsets"] is not None:
            offsets = np.asarray(skeleton_info["offsets"], dtype=np.float32)
        if "parent_indices" in skeleton_info and skeleton_info["parent_indices"] is not None:
            parents = list(skeleton_info["parent_indices"])

    if stream_type == "joint_pos":
        world_pos = compute_joint_positions(root_pos, joint_quats, offsets, parents)
        # Make root-relative by subtracting hips position
        joint_feats = world_pos - world_pos[:, 0:1, :]
        return _pack_25_node(root_pos, joint_feats), 3

    if stream_type == "bone":
        world_pos = compute_joint_positions(root_pos, joint_quats, offsets, parents)
        bones = compute_bone_vectors(world_pos, parents)
        return _pack_25_node(root_pos, bones), 3

    if stream_type == "joint_motion":
        world_pos = compute_joint_positions(root_pos, joint_quats, offsets, parents)
        joint_feats = world_pos - world_pos[:, 0:1, :]
        motion = compute_temporal_differences(joint_feats)
        return _pack_25_node(np.zeros_like(root_pos), motion), 3

    if stream_type == "bone_motion":
        world_pos = compute_joint_positions(root_pos, joint_quats, offsets, parents)
        bones = compute_bone_vectors(world_pos, parents)
        motion = compute_temporal_differences(bones)
        return _pack_25_node(np.zeros_like(root_pos), motion), 3

    raise ValueError(f"Unknown stream_type: {stream_type}")


# Stream → in_channels mapping for experiment configs
STREAM_CHANNELS: dict[str, int] = {
    "rotation_6d":  6,   # default, handled by pybvh_ml pipeline in MotionDataset
    "joint_pos":    3,
    "bone":         3,
    "joint_motion": 3,
    "bone_motion":  3,
}
