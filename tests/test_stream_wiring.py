"""TDD tests for stream_type wiring in exp034 and exp020 runners.

Phase ACC Track A: bone / bone_motion ensemble members.
See ~/.claude/plans/replicated-gathering-raven.md

Covers:
  - get_stream_features("bone") / ("bone_motion") shape and channel count
  - Node-0 semantics for bone (root_pos) and bone_motion (zero)
  - Node 1-24 content matches direct compute_bone_vectors / compute_temporal_differences
  - MotionDataModule.expected_in_channels regression guard
  - MotionDataModule constructor stores stream_type correctly
  - Real-data setup("fit") smoke when data is available
"""

from __future__ import annotations

import numpy as np
import pytest

from diema.features.multi_stream import (
    STREAM_CHANNELS,
    _DIEMA_PARENTS_BVH,
    compute_bone_vectors,
    compute_joint_positions,
    compute_temporal_differences,
    get_stream_features,
)
from diema.data.collate import MotionDataModule


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_seq():
    """Return (root_pos, joint_quats) with F=5 frames, 24 joints, identity quats."""
    F, J = 5, 24
    rng = np.random.default_rng(0)
    root_pos = rng.uniform(-1.0, 1.0, (F, 3)).astype(np.float32)
    joint_quats = np.zeros((F, J, 4), dtype=np.float32)
    joint_quats[..., 0] = 1.0  # identity quaternion (wxyz = [1,0,0,0])
    return root_pos, joint_quats


def _fake_split_dict(n_train=20, n_val=10):
    """Minimal split dict that MotionDataModule accepts."""
    train = [(f"t{i}.bvh", i) for i in range(n_train)]
    val = [(f"v{i}.bvh", n_train + i) for i in range(n_val)]
    return {"train": train, "val": val}


# ---------------------------------------------------------------------------
# 1. get_stream_features("bone") — shape and channel semantics
# ---------------------------------------------------------------------------

class TestGetStreamFeaturesBone:
    def test_output_shape_and_channels(self, tiny_seq):
        root_pos, quats = tiny_seq
        data_ctv, C = get_stream_features(root_pos, quats, None, "bone")
        assert data_ctv.shape == (3, 5, 25), f"expected (3,5,25), got {data_ctv.shape}"
        assert C == 3

    def test_dtype_is_float32(self, tiny_seq):
        root_pos, quats = tiny_seq
        data_ctv, _ = get_stream_features(root_pos, quats, None, "bone")
        assert data_ctv.dtype == np.float32

    def test_node0_equals_root_pos(self, tiny_seq):
        """Node 0 (CTV index 0) of the bone stream must store root_pos."""
        root_pos, quats = tiny_seq
        data_ctv, _ = get_stream_features(root_pos, quats, None, "bone")
        # data_ctv shape: (C=3, F=5, V=25); node 0 is data_ctv[:, :, 0]
        # root_pos shape: (F=5, 3) → transposed to (3, 5)
        np.testing.assert_allclose(
            data_ctv[:, :, 0],
            root_pos.T,
            atol=1e-5,
            err_msg="Node 0 of bone stream must equal root_pos",
        )

    def test_nodes1to24_match_compute_bone_vectors(self, tiny_seq):
        """Nodes 1-24 of the bone stream must match compute_bone_vectors output."""
        root_pos, quats = tiny_seq
        data_ctv, _ = get_stream_features(root_pos, quats, None, "bone")

        # Compute expected via direct functions
        world_pos = compute_joint_positions(root_pos, quats)       # (5, 24, 3)
        bones = compute_bone_vectors(world_pos)                    # (5, 24, 3)
        # Bones shape (F, 24, C) → packed to (C, F, 25); nodes 1-24 = joints 0-23
        # data_ctv[:, :, 1:] corresponds to joint_feats[:, :, 0:24]
        # which in (C,F,V) layout = bones.transpose(2,0,1)
        expected = bones.transpose(2, 0, 1)  # (3, 5, 24)
        np.testing.assert_allclose(
            data_ctv[:, :, 1:],
            expected,
            atol=1e-5,
            err_msg="Nodes 1-24 of bone stream must match compute_bone_vectors",
        )


# ---------------------------------------------------------------------------
# 2. get_stream_features("bone_motion") — shape and channel semantics
# ---------------------------------------------------------------------------

class TestGetStreamFeaturesBoneMotion:
    def test_output_shape_and_channels(self, tiny_seq):
        root_pos, quats = tiny_seq
        data_ctv, C = get_stream_features(root_pos, quats, None, "bone_motion")
        assert data_ctv.shape == (3, 5, 25), f"expected (3,5,25), got {data_ctv.shape}"
        assert C == 3

    def test_node0_is_zero(self, tiny_seq):
        """Node 0 (CTV index 0) of bone_motion must be zero (root is zeroed out)."""
        root_pos, quats = tiny_seq
        data_ctv, _ = get_stream_features(root_pos, quats, None, "bone_motion")
        np.testing.assert_allclose(
            data_ctv[:, :, 0],
            0.0,
            atol=1e-6,
            err_msg="Node 0 of bone_motion stream must be zero",
        )

    def test_nodes1to24_match_temporal_diff_of_bones(self, tiny_seq):
        """Nodes 1-24 must match compute_temporal_differences(bone_vectors)."""
        root_pos, quats = tiny_seq
        data_ctv, _ = get_stream_features(root_pos, quats, None, "bone_motion")

        world_pos = compute_joint_positions(root_pos, quats)   # (5, 24, 3)
        bones = compute_bone_vectors(world_pos)                 # (5, 24, 3)
        motion = compute_temporal_differences(bones)            # (5, 24, 3)
        expected = motion.transpose(2, 0, 1)                   # (3, 5, 24)
        np.testing.assert_allclose(
            data_ctv[:, :, 1:],
            expected,
            atol=1e-5,
            err_msg="Nodes 1-24 of bone_motion must match temporal diff of bones",
        )

    def test_first_frame_nodes1to24_is_zero(self, tiny_seq):
        """Temporal diff first frame is always zero-padded."""
        root_pos, quats = tiny_seq
        data_ctv, _ = get_stream_features(root_pos, quats, None, "bone_motion")
        # Frame index 0, all nodes except maybe root
        np.testing.assert_allclose(
            data_ctv[:, 0, 1:],
            0.0,
            atol=1e-6,
            err_msg="First frame nodes 1-24 of bone_motion must be zero (zero-padded diff)",
        )


# ---------------------------------------------------------------------------
# 3. MotionDataModule.expected_in_channels regression guard
# ---------------------------------------------------------------------------

class TestExpectedInChannels:
    def test_bone_is_3(self):
        assert MotionDataModule.expected_in_channels("bone") == 3

    def test_bone_motion_is_3(self):
        assert MotionDataModule.expected_in_channels("bone_motion") == 3

    def test_rotation_6d_is_6(self):
        """Regression guard: default stream must still report 6 channels."""
        assert MotionDataModule.expected_in_channels("rotation_6d") == 6

    def test_joint_pos_is_3(self):
        assert MotionDataModule.expected_in_channels("joint_pos") == 3

    def test_bone_with_no_lags_is_3(self):
        assert MotionDataModule.expected_in_channels("bone", None) == 3

    def test_bone_with_lags_multiplies(self):
        # 3 base × (1 + 2 lags) = 9
        assert MotionDataModule.expected_in_channels("bone", (1, 3)) == 9


# ---------------------------------------------------------------------------
# 4. MotionDataModule wiring — constructor stores stream_type
# ---------------------------------------------------------------------------

class TestDataModuleStreamWiring:
    def test_constructor_stores_bone(self, tmp_path):
        """MotionDataModule must expose the requested stream_type."""
        split = _fake_split_dict()
        dm = MotionDataModule(
            data_path=str(tmp_path / "fake.npz"),
            split_dict=split,
            stream_type="bone",
        )
        assert dm.stream_type == "bone"

    def test_constructor_stores_bone_motion(self, tmp_path):
        split = _fake_split_dict()
        dm = MotionDataModule(
            data_path=str(tmp_path / "fake.npz"),
            split_dict=split,
            stream_type="bone_motion",
        )
        assert dm.stream_type == "bone_motion"

    def test_default_stream_type_unchanged(self, tmp_path):
        """Default must remain rotation_6d so existing experiments are unaffected."""
        split = _fake_split_dict()
        dm = MotionDataModule(
            data_path=str(tmp_path / "fake.npz"),
            split_dict=split,
        )
        assert dm.stream_type == "rotation_6d"

    def test_setup_fit_propagates_stream_type_to_dataset(self, tmp_path):
        """After setup('fit'), the train dataset must carry the requested stream_type.

        Falls back to a safe assertion (constructor only) when real data is absent.
        """
        from utils.env import EnvConfig
        env = EnvConfig()
        real_data = env.processed_dir / "motion_train_quat.npz"

        split = _fake_split_dict(n_train=10, n_val=5)
        dm = MotionDataModule(
            data_path=str(real_data),
            split_dict=split,
            stream_type="bone",
            debug=True,
        )

        if not real_data.exists():
            # Data not available in this test environment — verify constructor only.
            assert dm.stream_type == "bone"
            pytest.skip("Real data not available; constructor-only check passed.")

        dm.setup("fit")
        assert dm.dataset_train.stream_type == "bone", (
            f"Expected dataset_train.stream_type='bone', "
            f"got '{dm.dataset_train.stream_type}'"
        )
