"""DIEM-A BVH-24 → PoseC3D COCO-17 heatmap-volume projection (exp086).

PoseC3D (PYSKL slowonly_r50_ntu60_xsub/joint.py) consumes Gaussian heatmap
volumes (17 COCO keypoints × T × H × W) rather than coordinates. This module
reproduces the PYSKL val_pipeline for that config:

  1. FK on BVH-24 → (F, 24, 3) world positions
     (reuses multi_stream.compute_joint_positions, like bvh_to_h36m17.py)
  2. Orthographic 2D project (drop depth "y", flip vertical) → (F, 24, 2)
     (same face-front convention validated for exp081 image-plane transfer)
  3. BVH-24 → COCO-17 (face joints 0-4 synthesized from Head w/ low conf —
     DIEM-A mocap has no face; body joints 5-16 carry the signal)
  4. UniformSampleFrames clip_len=48
  5. PoseCompact: tight bbox of all kp, pad 0.25, square (hw_ratio=1)
  6. Resize to 64×64
  7. GeneratePoseTarget: per-joint Gaussian, sigma=0.6,
        patch = exp(-((x-mu_x)^2+(y-mu_y)^2)/2/sigma^2) * conf,
     accumulated with np.maximum  (exact PYSKL formula)
  → (17, 48, 64, 64) float32 heatmap volume per clip.

Reference (not imported): tmp/pyskl heatmap_related.py / augmentations.py
Related: diema/features/bvh_to_h36m17.py, diema/features/bvh_to_ntu25.py
"""

from __future__ import annotations

from typing import Any

import numpy as np

from diema.features.multi_stream import compute_joint_positions

# COCO-17 ← BVH-24 (None → synthesized from Head with low confidence)
COCO17_TO_BVH: dict[int, int | str] = {
    0: "nose",        # ← Head, conf 0.5
    1: "leye",        # ← Head + small offset, conf 0.3
    2: "reye",        # ← Head + small offset, conf 0.3
    3: "lear",        # ← Head, conf 0.3
    4: "rear",        # ← Head, conf 0.3
    5: 13,            # left_shoulder ← LeftArm
    6: 9,             # right_shoulder ← RightArm
    7: 14,            # left_elbow ← LeftForeArm
    8: 10,            # right_elbow ← RightForeArm
    9: 15,            # left_wrist ← LeftHand
    10: 11,           # right_wrist ← RightHand
    11: 20,           # left_hip ← LeftUpLeg
    12: 16,           # right_hip ← RightUpLeg
    13: 21,           # left_knee ← LeftLeg
    14: 17,           # right_knee ← RightLeg
    15: 22,           # left_ankle ← LeftFoot
    16: 18,           # right_ankle ← RightFoot
}
_BVH_HEAD = 7
EPS = 1e-4


def _uniform_sample(num_frames: int, clip_len: int) -> np.ndarray:
    if num_frames <= 0 or clip_len <= 0:
        raise ValueError(f"num_frames {num_frames}, clip_len {clip_len}")
    if num_frames == 1:
        return np.zeros(clip_len, dtype=np.int64)
    return np.round(np.linspace(0, num_frames - 1, clip_len)).astype(np.int64)


def _bvh24_to_coco17(xy24: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T,24,2) → (T,17,2) coords + (T,17) confidence."""
    T = xy24.shape[0]
    out = np.zeros((T, 17, 2), dtype=np.float32)
    conf = np.ones((T, 17), dtype=np.float32)
    head = xy24[:, _BVH_HEAD, :]
    # rough head extent for synthetic eye/ear offsets (per-frame torso scale)
    span = np.linalg.norm(xy24.max(axis=1) - xy24.min(axis=1), axis=-1, keepdims=True)
    off = 0.03 * np.maximum(span, 1e-6)
    for c, src in COCO17_TO_BVH.items():
        if isinstance(src, int):
            out[:, c, :] = xy24[:, src, :]
        elif src == "nose":
            out[:, c, :] = head
            conf[:, c] = 0.5
        elif src == "leye":
            out[:, c, :] = head + np.concatenate([+off, +off], axis=-1)
            conf[:, c] = 0.3
        elif src == "reye":
            out[:, c, :] = head + np.concatenate([-off, +off], axis=-1)
            conf[:, c] = 0.3
        elif src == "lear":
            out[:, c, :] = head + np.concatenate([+2 * off, np.zeros_like(off)], axis=-1)
            conf[:, c] = 0.3
        elif src == "rear":
            out[:, c, :] = head + np.concatenate([-2 * off, np.zeros_like(off)], axis=-1)
            conf[:, c] = 0.3
    return out, conf


def _pose_compact_resize(kp: np.ndarray, padding: float, out_hw: int) -> np.ndarray:
    """kp (T,17,2) → coords mapped into [0,out_hw) square box (PoseCompact +
    Resize, hw_ratio=1, allow_imgpad=True)."""
    nz = kp[(kp[..., 0] != 0) | (kp[..., 1] != 0)]
    if nz.size == 0:
        return np.zeros_like(kp)
    min_x, min_y = nz[:, 0].min(), nz[:, 1].min()
    max_x, max_y = nz[:, 0].max(), nz[:, 1].max()
    cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    half_w = (max_x - min_x) / 2.0 * (1.0 + padding)
    half_h = (max_y - min_y) / 2.0 * (1.0 + padding)
    half = max(half_w, half_h, 1e-3)  # square (hw_ratio=1)
    box_min_x, box_min_y = cx - half, cy - half
    scale = (out_hw - 1) / (2.0 * half)
    out = kp.copy().astype(np.float32)
    out[..., 0] = (out[..., 0] - box_min_x) * scale
    out[..., 1] = (out[..., 1] - box_min_y) * scale
    return out


def _gaussian_volume(kp: np.ndarray, conf: np.ndarray, hw: int, sigma: float) -> np.ndarray:
    """kp (T,17,2) in [0,hw), conf (T,17) → (17,T,hw,hw) heatmap volume.

    Exact PYSKL formula: exp(-((x-mux)^2+(y-muy)^2)/2/sigma^2)*conf, with a
    3*sigma support window (max-accumulated; here one person so plain set).
    """
    T, V, _ = kp.shape
    vol = np.zeros((V, T, hw, hw), dtype=np.float32)
    rad = max(int(3 * sigma) + 1, 1)
    for t in range(T):
        for j in range(V):
            cval = conf[t, j]
            if cval < EPS:
                continue
            mx, my = kp[t, j, 0], kp[t, j, 1]
            st_x = max(int(mx - 3 * sigma), 0)
            ed_x = min(int(mx + 3 * sigma) + 1, hw)
            st_y = max(int(my - 3 * sigma), 0)
            ed_y = min(int(my + 3 * sigma) + 1, hw)
            if ed_x <= st_x or ed_y <= st_y:
                continue
            xs = np.arange(st_x, ed_x, dtype=np.float32)
            ys = np.arange(st_y, ed_y, dtype=np.float32)[:, None]
            patch = np.exp(-((xs - mx) ** 2 + (ys - my) ** 2) / 2.0 / sigma ** 2) * cval
            vol[j, t, st_y:ed_y, st_x:ed_x] = np.maximum(
                vol[j, t, st_y:ed_y, st_x:ed_x], patch
            )
    return vol


def project_bvh_to_coco17_heatmap(
    root_pos: np.ndarray,
    joint_quats: np.ndarray,
    clip_len: int = 48,
    hw: int = 64,
    sigma: float = 0.6,
    padding: float = 0.25,
    skeleton_info: dict[str, Any] | None = None,
    drop_axis: str = "y",
    flip_vertical: bool = True,
) -> np.ndarray:
    """One DIEM-A clip → (17, clip_len, hw, hw) float32 heatmap volume."""
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must be (F,3), got {root_pos.shape}")
    if joint_quats.ndim != 3 or joint_quats.shape[1:] != (24, 4):
        raise ValueError(f"joint_quats must be (F,24,4), got {joint_quats.shape}")
    if root_pos.shape[0] != joint_quats.shape[0]:
        raise ValueError("root_pos and joint_quats frame count differ")

    world = compute_joint_positions(
        root_pos, joint_quats,
        offsets=None if skeleton_info is None else skeleton_info.get("offsets"),
        parents=None if skeleton_info is None else skeleton_info.get("parent_indices"),
    )  # (F,24,3)
    if drop_axis == "y":
        xy = world[:, :, [0, 2]].copy()
    elif drop_axis == "z":
        xy = world[:, :, [0, 1]].copy()
    elif drop_axis == "x":
        xy = world[:, :, [1, 2]].copy()
    else:
        raise ValueError(f"drop_axis must be x|y|z, got {drop_axis!r}")
    if flip_vertical:
        xy[:, :, 1] = -xy[:, :, 1]

    idx = _uniform_sample(xy.shape[0], clip_len)
    xy = xy[idx]  # (clip_len, 24, 2)
    coco, conf = _bvh24_to_coco17(xy)
    coco = _pose_compact_resize(coco, padding=padding, out_hw=hw)
    return _gaussian_volume(coco, conf, hw=hw, sigma=sigma)
