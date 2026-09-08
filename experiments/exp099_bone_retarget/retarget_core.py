"""retarget_core.py — Stage 2 of exp099 bone-retarget pipeline.

Implements bone-length scaling retarget via forward kinematics (FK).

Math (per prompt §STAGE 2):
1. Source joint rotations `joint_quats` (T, 24, 4) and root translation
   `root_pos` (T, 3) carry the motion intent (preserved).
2. Source offsets `source_offsets` (24, 3) and target offsets
   `target_offsets` (24, 3) carry the rest-pose bone geometry.
3. Retarget = run FK with **target** offsets while keeping rotations.
   Direction is preserved from target's rest pose (per-performer
   variation). New joint positions = different bone geometry, same
   motion intent.
4. Optional ground-contact correction: shift `root_pos` along the
   world vertical axis (Z, axis 2) so the lowest world joint at frame 0
   lands at the original (source) ground level.

Why bone-length scaling only (no rotation modification)?
- Rotations encode the performer's motion intent (anger, sadness, etc.)
  which is the emotion signal we want to preserve.
- Bone proportions encode the performer's body shape, which is the
  style nuisance variable we want to perturb.

Caveats (documented):
- Rotation-only model inputs (e.g. SkateFormer's rotation_6d) are
  unaffected by this retarget (rotations are scale-invariant).
  training must use joint_pos input or re-derive joint positions per
  clip with retargeted offsets to actually test the hypothesis.
- BVH-24 indexing: this module operates in BVH-24 joint space (root
  at index 0). CTV-25 packing (virtual_root + 24 BVH joints) happens
  downstream in `diema/models/skeleton_graph.py`. The unit test
  `test_bvh_ctv_indexing` documents the boundary.

Related: `diema/features/multi_stream.py::compute_joint_positions` is
the FK kernel reused here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Reuse the production FK kernel (BVH-24 space).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from diema.features.multi_stream import (  # noqa: E402
    compute_joint_positions, _DIEMA_PARENTS_BVH)

# World-frame vertical (up) axis index. The DIEM-A skeleton is Z-up in
# world space: empirically the head→foot vector is ~[−9, +2, +58] (axis 2
# dominates), and a true vertical-axis rotation preserves the axis-2
# component. Earlier code used axis 1 (Y) for ground contact, which is a
# horizontal axis — see docs/analysis for the up-axis investigation.
_UP_AXIS = 2


def retarget_clip(
    joint_quats: np.ndarray,        # (T, 24, 4) wxyz
    root_pos: np.ndarray,            # (T, 3) world root position
    source_offsets: np.ndarray,      # (24, 3) source performer's offsets
    target_offsets: np.ndarray,      # (24, 3) target performer's offsets
    parents: list[int] | None = None,
    preserve_ground_contact: bool = True,
) -> dict:
    """Apply bone-length retarget to a motion clip.

    Parameters
    ----------
    joint_quats : (T, 24, 4)
        Per-frame joint rotations as quaternions (w, x, y, z) in local
        parent frame. Preserved unchanged in output.
    root_pos : (T, 3)
        Per-frame world-frame root (Hips) translation.
    source_offsets : (24, 3)
        The source performer's rest-pose bone offsets (BVH header).
        Used only for ground-contact reference (frame 0 lowest Z).
    target_offsets : (24, 3)
        The target performer's rest-pose bone offsets. FK uses these
        as the new bone geometry.
    parents : list of 24 ints, optional
        Parent index per joint; root has -1. Defaults to
        `_DIEMA_PARENTS_BVH`.
    preserve_ground_contact : bool
        If True, shift root along the world vertical axis (Z) so the
        lowest joint at frame 0 retains its original world Z.

    Returns
    -------
    dict with keys:
        joint_positions : (T, 24, 3) world coords with target geometry
        new_root_pos    : (T, 3) updated root translation
        rotation_quat   : (T, 24, 4) same as input (for convenience)
        new_offsets     : (24, 3) = target_offsets (for downstream use)
        meta : dict with summary stats
    """
    if joint_quats.ndim != 3 or joint_quats.shape[1:] != (24, 4):
        raise ValueError(
            f"joint_quats must be (T, 24, 4); got {joint_quats.shape}")
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(
            f"root_pos must be (T, 3); got {root_pos.shape}")
    if joint_quats.shape[0] != root_pos.shape[0]:
        raise ValueError(
            f"T mismatch: joint_quats {joint_quats.shape[0]} vs "
            f"root_pos {root_pos.shape[0]}")
    if source_offsets.shape != (24, 3):
        raise ValueError(
            f"source_offsets must be (24, 3); got {source_offsets.shape}")
    if target_offsets.shape != (24, 3):
        raise ValueError(
            f"target_offsets must be (24, 3); got {target_offsets.shape}")

    parents = parents if parents is not None else _DIEMA_PARENTS_BVH
    assert len(parents) == 24, f"parents must have 24 elements, got {len(parents)}"

    T = joint_quats.shape[0]
    # FK with target offsets
    tgt_pos = compute_joint_positions(
        root_pos=root_pos.astype(np.float32),
        joint_quats=joint_quats.astype(np.float32),
        offsets=target_offsets.astype(np.float32),
        parents=parents,
    )
    # tgt_pos shape: (T, 24, 3)

    new_root_pos = root_pos.copy().astype(np.float32)
    if preserve_ground_contact and T > 0:
        # Compute source frame-0 lowest vertical coord for reference.
        # _UP_AXIS == 2 (Z) — the world-frame vertical axis for DIEM-A.
        src_pos = compute_joint_positions(
            root_pos=root_pos.astype(np.float32),
            joint_quats=joint_quats.astype(np.float32),
            offsets=source_offsets.astype(np.float32),
            parents=parents,
        )
        src_min_up = float(src_pos[0, :, _UP_AXIS].min())
        tgt_min_up = float(tgt_pos[0, :, _UP_AXIS].min())
        delta_up = src_min_up - tgt_min_up
        new_root_pos[:, _UP_AXIS] += delta_up
        tgt_pos = tgt_pos.copy()
        tgt_pos[:, :, _UP_AXIS] += delta_up
    else:
        delta_up = 0.0

    # bone-length stats (for diagnostics)
    src_lens = np.linalg.norm(source_offsets, axis=1)
    tgt_lens = np.linalg.norm(target_offsets, axis=1)
    non_root = src_lens > 1e-6
    rel_dev = np.zeros_like(src_lens)
    rel_dev[non_root] = (np.abs(tgt_lens - src_lens)[non_root] /
                          src_lens[non_root])

    return {
        "joint_positions": tgt_pos.astype(np.float32),
        "new_root_pos": new_root_pos,
        "rotation_quat": joint_quats.astype(np.float32),
        "new_offsets": target_offsets.astype(np.float32),
        "meta": {
            "T": int(T),
            "delta_up": float(delta_up),
            "up_axis": _UP_AXIS,
            "bone_length_max_rel_dev": float(rel_dev.max()),
            "bone_length_mean_rel_dev": float(rel_dev[non_root].mean()),
            "preserve_ground_contact": bool(preserve_ground_contact),
        },
    }


# ---------------------------------------------------------------------------
# Helpers — bone-index utilities (CTV vs BVH boundary)
# ---------------------------------------------------------------------------

def bvh24_bone_pairs() -> list[tuple[int, int]]:
    """Return list of (parent, child) bone index pairs in BVH-24 space.
    Index 0 (root/Hips) has parent -1 and is excluded.
    """
    return [(_DIEMA_PARENTS_BVH[j], j)
             for j in range(24)
             if _DIEMA_PARENTS_BVH[j] >= 0]


def bvh24_to_ctv25_index(bvh_idx: int) -> int:
    """Translate a BVH-24 joint index to its CTV-25 node index.
    CTV node 0 = virtual_root; CTV nodes 1-24 = BVH joints 0-23 shifted
    by +1. So BVH idx j maps to CTV idx j+1.
    """
    if not (0 <= bvh_idx < 24):
        raise ValueError(f"bvh_idx must be in [0, 24); got {bvh_idx}")
    return bvh_idx + 1
