"""Tests for diema/features/multi_stream.py."""

import numpy as np
import pytest

from diema.features.multi_stream import (
    STREAM_CHANNELS,
    _DIEMA_PARENTS_BVH,
    _load_reference_offsets,
    _quat_multiply,
    _quat_rotate_vector,
    compute_bone_vectors,
    compute_joint_positions,
    compute_temporal_differences,
    get_stream_features,
)


@pytest.fixture
def dummy_sequence():
    """Return (root_pos, joint_quats) for a 10-frame identity-rotation sequence."""
    F, J = 10, 24
    root_pos = np.zeros((F, 3), dtype=np.float32)
    root_pos[:, 2] = np.arange(F, dtype=np.float32)  # move along Z
    joint_quats = np.zeros((F, J, 4), dtype=np.float32)
    joint_quats[..., 0] = 1.0  # identity quaternion (wxyz)
    return root_pos, joint_quats


class TestQuaternionOps:
    def test_quat_multiply_identity(self):
        ident = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        q = np.array([[0.707, 0.707, 0.0, 0.0]], dtype=np.float32)
        out = _quat_multiply(ident, q)
        np.testing.assert_allclose(out, q, atol=1e-6)

    def test_quat_rotate_identity(self):
        ident = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = _quat_rotate_vector(ident, v)
        np.testing.assert_allclose(out, [[1, 2, 3]], atol=1e-6)

    def test_quat_rotate_90_around_z(self):
        # 90° around Z axis: (cos45, 0, 0, sin45)
        import math
        s = math.sin(math.pi / 4)
        q = np.array([[math.cos(math.pi / 4), 0, 0, s]], dtype=np.float32)
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        out = _quat_rotate_vector(q, v)
        # Rotating +X by 90° around Z gives +Y
        np.testing.assert_allclose(out, [[0, 1, 0]], atol=1e-5)


class TestForwardKinematics:
    def test_identity_rotations_give_rest_pose(self, dummy_sequence):
        root_pos, joint_quats = dummy_sequence
        offsets = _load_reference_offsets()
        parents = _DIEMA_PARENTS_BVH
        positions = compute_joint_positions(root_pos, joint_quats, offsets, parents)

        # Hips (node 0) should equal root_pos
        np.testing.assert_allclose(positions[:, 0], root_pos, atol=1e-5)

        # Spine (node 1) = Hips + offset[1] at each frame
        expected_spine = root_pos + offsets[1][None, :]
        np.testing.assert_allclose(positions[:, 1], expected_spine, atol=1e-5)

        # RightHand (node 11) — should be Hips + cumulative offsets along the arm chain
        chain = [0, 1, 2, 3, 4, 8, 9, 10, 11]  # parent chain to RightHand
        cumulative = sum(offsets[c] for c in chain[1:])  # skip Hips offset (zero)
        expected = np.broadcast_to(cumulative, (10, 3))
        np.testing.assert_allclose(positions[:, 11] - root_pos, expected, atol=1e-5)

    def test_positions_shape(self, dummy_sequence):
        root_pos, joint_quats = dummy_sequence
        positions = compute_joint_positions(root_pos, joint_quats)
        assert positions.shape == (10, 24, 3)

    def test_bone_vectors(self, dummy_sequence):
        root_pos, joint_quats = dummy_sequence
        positions = compute_joint_positions(root_pos, joint_quats)
        bones = compute_bone_vectors(positions)
        assert bones.shape == (10, 24, 3)
        # Root bone is zero
        np.testing.assert_allclose(bones[:, 0], 0.0, atol=1e-6)
        # Spine bone should equal offset[1]
        offsets = _load_reference_offsets()
        expected = np.broadcast_to(offsets[1], (10, 3))
        np.testing.assert_allclose(bones[:, 1], expected, atol=1e-5)


class TestTemporalDifferences:
    def test_first_frame_zero(self):
        x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        diff = compute_temporal_differences(x)
        np.testing.assert_allclose(diff[0], 0.0)
        np.testing.assert_allclose(diff[1], [2, 2])
        np.testing.assert_allclose(diff[2], [2, 2])


class TestGetStreamFeatures:
    @pytest.mark.parametrize("stream", ["joint_pos", "bone", "joint_motion", "bone_motion"])
    def test_output_shape(self, dummy_sequence, stream):
        root_pos, joint_quats = dummy_sequence
        data_ctv, C = get_stream_features(root_pos, joint_quats, stream_type=stream)
        assert data_ctv.shape == (3, 10, 25)
        assert C == 3
        assert data_ctv.dtype == np.float32

    def test_joint_pos_node0_is_root(self, dummy_sequence):
        root_pos, joint_quats = dummy_sequence
        data_ctv, _ = get_stream_features(root_pos, joint_quats, stream_type="joint_pos")
        # Node 0 should hold root_pos channels
        np.testing.assert_allclose(data_ctv[:, :, 0], root_pos.T, atol=1e-6)

    def test_joint_pos_hips_is_zero(self, dummy_sequence):
        # With root-relative positions, Hips (CTV node 1) should be zero
        root_pos, joint_quats = dummy_sequence
        data_ctv, _ = get_stream_features(root_pos, joint_quats, stream_type="joint_pos")
        np.testing.assert_allclose(data_ctv[:, :, 1], 0.0, atol=1e-5)

    def test_unknown_stream_raises(self, dummy_sequence):
        root_pos, joint_quats = dummy_sequence
        with pytest.raises(ValueError):
            get_stream_features(root_pos, joint_quats, stream_type="bogus")


class TestStreamChannels:
    def test_channel_mapping(self):
        assert STREAM_CHANNELS["rotation_6d"] == 6
        assert STREAM_CHANNELS["joint_pos"] == 3
        assert STREAM_CHANNELS["bone"] == 3
        assert STREAM_CHANNELS["joint_motion"] == 3
        assert STREAM_CHANNELS["bone_motion"] == 3
