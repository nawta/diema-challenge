"""Unit tests for retarget_core (Stage 2 of exp099 bone-retarget).

Five tests per prompt §STAGE 2:
1. test_identity_retarget   — same offsets in → joint pos exactly equal
2. test_scaling_changes_bones — 2x bone offset → joint pos moves 2x
3. test_rotations_preserved — rotation_quat output == input
4. test_root_pos_preserved  — root x/y trajectory unchanged (Z may shift)
5. test_bvh_ctv_indexing    — bone & CTV index helpers correct

Run: conda run -n acii2026 pytest \
        experiments/exp099_bone_retarget/test_retarget_core.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retarget_core import (
    retarget_clip, bvh24_bone_pairs, bvh24_to_ctv25_index)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from diema.features.multi_stream import (
    _load_reference_offsets, _DIEMA_PARENTS_BVH, compute_joint_positions)
from diema.models.skeleton_graph import DIEMA_INWARD_EDGES


def _rand_clip(T=20, rng=None):
    rng = rng or np.random.default_rng(42)
    # Random unit quaternions (T, 24, 4) — slightly biased toward identity
    q = rng.normal(size=(T, 24, 4)).astype(np.float32)
    q[..., 0] += 2.0  # bias w-component → near-identity rotations
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    root_pos = rng.normal(size=(T, 3)).astype(np.float32) * 5.0
    return q, root_pos


# --- Test 1 ---------------------------------------------------------

def test_identity_retarget():
    """Same source/target offsets → joint positions unchanged."""
    q, root = _rand_clip(T=10)
    offsets = _load_reference_offsets()
    out = retarget_clip(
        joint_quats=q, root_pos=root,
        source_offsets=offsets, target_offsets=offsets,
        preserve_ground_contact=False,
    )
    # FK with identical offsets must be exactly equal
    ref_pos = compute_joint_positions(
        root_pos=root, joint_quats=q, offsets=offsets,
        parents=_DIEMA_PARENTS_BVH,
    )
    np.testing.assert_allclose(out["joint_positions"], ref_pos,
                                rtol=1e-6, atol=1e-6)
    # delta_up stays exactly 0 when ground-contact preservation is off
    assert out["meta"]["delta_up"] == 0.0
    assert out["meta"]["bone_length_max_rel_dev"] == 0.0


# --- Test 2 ---------------------------------------------------------

def test_scaling_changes_bones():
    """Doubling target bone offsets → joint positions move 2x relative
    to root (modulo direction-of-rotation; verified via bone-length
    invariant)."""
    q, root = _rand_clip(T=5)
    offsets = _load_reference_offsets()
    target = offsets * 2.0  # uniform 2x scale

    out = retarget_clip(
        joint_quats=q, root_pos=root,
        source_offsets=offsets, target_offsets=target,
        preserve_ground_contact=False,
    )
    new_pos = out["joint_positions"]

    # For each bone (parent, child), the bone vector length in the new
    # frame should be exactly 2x the bone vector length in the source
    # frame (regardless of rotations, by isotropic scaling property).
    src_pos = compute_joint_positions(
        root_pos=root, joint_quats=q, offsets=offsets,
        parents=_DIEMA_PARENTS_BVH,
    )
    for parent_idx, child_idx in bvh24_bone_pairs():
        src_len = np.linalg.norm(
            src_pos[:, child_idx, :] - src_pos[:, parent_idx, :], axis=-1)
        new_len = np.linalg.norm(
            new_pos[:, child_idx, :] - new_pos[:, parent_idx, :], axis=-1)
        # if src bone length > 0, ratio should be 2.0
        mask = src_len > 1e-3
        if mask.any():
            ratios = new_len[mask] / src_len[mask]
            np.testing.assert_allclose(ratios, 2.0, rtol=1e-4)


# --- Test 3 ---------------------------------------------------------

def test_rotations_preserved():
    """Output rotation_quat must equal input element-wise."""
    q, root = _rand_clip(T=8)
    offsets = _load_reference_offsets()
    target = offsets * 1.3
    out = retarget_clip(
        joint_quats=q, root_pos=root,
        source_offsets=offsets, target_offsets=target,
        preserve_ground_contact=True,
    )
    np.testing.assert_array_equal(out["rotation_quat"], q.astype(np.float32))


# --- Test 4 ---------------------------------------------------------

def test_root_pos_preserved():
    """Root x/y trajectory must be unchanged regardless of ground-
    contact preservation. Only the vertical axis (Z) may shift
    (uniformly across time)."""
    q, root = _rand_clip(T=12)
    offsets = _load_reference_offsets()
    target = offsets * 0.85  # shorter
    out = retarget_clip(
        joint_quats=q, root_pos=root,
        source_offsets=offsets, target_offsets=target,
        preserve_ground_contact=True,
    )
    new_root = out["new_root_pos"]
    # x and y (the two ground-plane axes) trajectories unchanged
    np.testing.assert_array_equal(new_root[:, 0], root[:, 0].astype(np.float32))
    np.testing.assert_array_equal(new_root[:, 1], root[:, 1].astype(np.float32))
    # z (vertical) has a uniform shift across all frames
    delta_up_per_frame = new_root[:, 2] - root[:, 2].astype(np.float32)
    np.testing.assert_allclose(delta_up_per_frame,
                                delta_up_per_frame[0] * np.ones_like(delta_up_per_frame),
                                rtol=1e-5, atol=1e-5)
    # And the shift equals what meta reports
    np.testing.assert_allclose(delta_up_per_frame[0],
                                out["meta"]["delta_up"], rtol=1e-5)


# --- Test 5 ---------------------------------------------------------

def test_bvh_ctv_indexing():
    """Verify bone-index helpers map correctly between BVH-24 and
    CTV-25, and that bone pairs match DIEMA_INWARD_EDGES under the
    +1 shift (CTV node 0 = virtual_root; CTV nodes 1-24 = BVH 0-23)."""
    # Helper round-trip
    for bvh in range(24):
        ctv = bvh24_to_ctv25_index(bvh)
        assert ctv == bvh + 1, f"bvh {bvh} → ctv {ctv}, expected {bvh+1}"

    # bvh24_bone_pairs must match DIEMA_INWARD_EDGES under +1 shift
    bvh_pairs = bvh24_bone_pairs()
    assert len(bvh_pairs) == 23, \
        f"expected 23 non-root bones, got {len(bvh_pairs)}"
    for parent_bvh, child_bvh in bvh_pairs:
        parent_ctv = bvh24_to_ctv25_index(parent_bvh)
        child_ctv = bvh24_to_ctv25_index(child_bvh)
        assert (child_ctv, parent_ctv) in DIEMA_INWARD_EDGES \
            or (parent_ctv, child_ctv) in DIEMA_INWARD_EDGES, \
            (f"BVH bone ({parent_bvh},{child_bvh}) -> CTV "
              f"({parent_ctv},{child_ctv}) not in DIEMA_INWARD_EDGES")

    # Out-of-range guard
    with pytest.raises(ValueError):
        bvh24_to_ctv25_index(24)
    with pytest.raises(ValueError):
        bvh24_to_ctv25_index(-1)
