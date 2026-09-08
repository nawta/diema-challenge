"""PyTorch Dataset for preprocessed motion capture sequences.

Loads quaternion data from pybvh-ml's NPZ format, applies augmentation
in quaternion space, converts to target representation, and packs to
(C, T, V) tensors for the model.

See: diema_challenge_implementation_spec.md §6.3 — データローダ
Related: diema/data/collate.py (batch collation), diema/features/motion_repr.py

Ported from: an internal baseline (Feeder class) with changes:
  - Hydra config compatible
  - Test mode label handling (label-less for hidden test)
  - Enhanced metadata return
"""

import numpy as np
import torch
import torch.utils.data

import pybvh_ml


class MotionDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for preprocessed skeleton sequences.

    Supports multiple input streams via the `stream_type` parameter:
      - "rotation_6d" (default): pybvh_ml pipeline (quaternion → 6D → pack_to_ctv)
      - "joint_pos":  root-relative 3D positions (forward kinematics)
      - "bone":       child - parent position vectors
      - "joint_motion": frame-to-frame differences of joint_pos
      - "bone_motion":  frame-to-frame differences of bone

    Pipeline per sample (rotation_6d):
      1. Load quaternion data from NPZ (root_pos + joint_quats)
      2. Augment in quaternion space (train mode only)
      3. Convert to target representation (6D, Euler, etc.)
      4. Pack to (C, T, V) layout via pybvh_ml.pack_to_ctv
      5. Temporal sample to fixed clip_length

    Pipeline per sample (joint_pos/bone/motion):
      1. Load quaternion data from NPZ
      2. Augment in quaternion space (train mode only)
      3. Run forward kinematics to get joint_positions
      4. Compute stream-specific features (positions / bones / diffs)
      5. Pack to (C, T, 25) with root as node 0
      6. Temporal sample

    Args:
        data_path: path to the .npz file (pybvh-ml format)
        indices: optional list of clip indices to select a subset
        clip_length: number of frames to sample per sequence
        target_repr: target rotation representation (used for rotation_6d stream)
            ('euler', '6d', 'quaternion', 'axisangle', 'rotmat')
        is_test: if True, use deterministic (center) sampling
        seed: random seed for reproducibility
        augmentation_pipeline: optional pybvh_ml.AugmentationPipeline
            (only applied when is_test=False)
        euler_orders: per-joint Euler orders (required for target_repr='euler')
        stream_type: which feature stream to compute (default: "rotation_6d")
        sampling_mode: "uniform" or "trimmed_uniform" (excludes head/tail 10%)
    """

    def __init__(
        self,
        data_path: str,
        indices: list[int] | None = None,
        clip_length: int = 64,
        target_repr: str = "6d",
        is_test: bool = False,
        seed: int = 255,
        augmentation_pipeline=None,
        euler_orders=None,
        stream_type: str = "rotation_6d",
        sampling_mode: str = "uniform",
        joint_dropout_prob: float = 0.0,
        # sequence cutout per body-part (Darragh 8th place ASL Signs)
        seq_cutout_p: float = 0.0,
        seq_cutout_segments: int = 5,
        seq_cutout_ratio: float = 0.15,
        # stack lag-difference features after sampling
        time_shift_lags: tuple[int, ...] | None = None,
        # per-sample isotropic body-scale jitter (positional channels only)
        body_scale_jitter_range: tuple[float, float] | None = None,
    ):
        preprocessed = pybvh_ml.load_preprocessed(data_path)
        self.clips = preprocessed["clips"]
        self.labels = preprocessed.get("labels")
        self.filenames = preprocessed.get("filenames", [])
        self.skeleton_info = preprocessed.get("skeleton_info", {})

        if indices is not None:
            self.clips = [self.clips[i] for i in indices]
            if self.labels is not None:
                self.labels = self.labels[indices]
            if self.filenames:
                self.filenames = [self.filenames[i] for i in indices]

        self.clip_length = clip_length
        self.target_repr = target_repr
        self.mode = "test" if is_test else "train"
        self.pipeline = augmentation_pipeline
        self._base_seed = seed
        self._rng = np.random.default_rng(seed)

        self.euler_orders = euler_orders or self.skeleton_info.get("euler_orders")

        # Multi-stream support
        self.stream_type = stream_type
        self.sampling_mode = sampling_mode
        self.joint_dropout_prob = joint_dropout_prob
        if stream_type not in ("rotation_6d", "joint_pos", "bone", "joint_motion", "bone_motion"):
            raise ValueError(f"Unknown stream_type: {stream_type}")
        if sampling_mode not in ("uniform", "trimmed_uniform"):
            raise ValueError(f"Unknown sampling_mode: {sampling_mode}")
        if not (0.0 <= joint_dropout_prob <= 1.0):
            raise ValueError(f"joint_dropout_prob must be in [0, 1], got {joint_dropout_prob}")

        # augmentations (sequence cutout per body-part + time deltas)
        if not (0.0 <= seq_cutout_p <= 1.0):
            raise ValueError(f"seq_cutout_p must be in [0, 1], got {seq_cutout_p}")
        self.seq_cutout_p = seq_cutout_p
        self.seq_cutout_segments = seq_cutout_segments
        self.seq_cutout_ratio = seq_cutout_ratio
        self.time_shift_lags = tuple(time_shift_lags) if time_shift_lags else None

        # body-scale jitter: validate and stash range
        if body_scale_jitter_range is not None:
            lo, hi = body_scale_jitter_range
            if not (0.0 < lo and lo <= hi):
                raise ValueError(
                    f"body_scale_jitter_range must satisfy 0 < lo <= hi, got {body_scale_jitter_range}"
                )
        self.body_scale_jitter_range = (
            tuple(body_scale_jitter_range) if body_scale_jitter_range else None
        )

        # Body-part groups for joint dropout (CTV node indices)
        from diema.models.skeleton_graph import DIEMA_BODY_PARTS
        self._body_parts = list(DIEMA_BODY_PARTS.values())
        self._body_parts_dict = dict(DIEMA_BODY_PARTS)

    def reseed_rng(self, worker_id: int) -> None:
        """Re-seed the augmentation RNG for a specific DataLoader worker.

        Called by DataLoader worker_init_fn so that each worker uses a
        distinct random stream, avoiding identical augmentations.
        """
        self._rng = np.random.default_rng(self._base_seed + worker_id)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        clip = self.clips[idx]
        root_pos = clip["root_pos"].copy()      # (F, 3)
        joint_data = clip["joint_data"].copy()   # (F, J, 4) quaternions

        # 1. Augment in quaternion space (train mode only)
        if self.pipeline is not None and self.mode == "train":
            joint_data, root_pos = self.pipeline(joint_data, root_pos, rng=self._rng)

        # 2. Build stream features
        if self.stream_type == "rotation_6d":
            # Legacy pipeline: convert to target_repr then pack via pybvh_ml
            if self.target_repr != "quaternion":
                stream_data = pybvh_ml.convert_arrays(
                    joint_data, "quaternion", self.target_repr,
                    euler_orders=self.euler_orders,
                )
            else:
                stream_data = joint_data
            data_ctv = pybvh_ml.pack_to_ctv(root_pos, stream_data, center_root=False)
        else:
            # FK-based stream (joint_pos / bone / joint_motion / bone_motion)
            from diema.features.multi_stream import get_stream_features
            data_ctv, _ = get_stream_features(
                root_pos, joint_data,
                skeleton_info=self.skeleton_info,
                stream_type=self.stream_type,
            )

        # 3. Temporal sample
        num_frames = data_ctv.shape[1]
        if self.sampling_mode == "trimmed_uniform":
            # Trim head/tail 10% then sample uniformly within that window.
            trim = max(int(num_frames * 0.1), 0)
            head = trim
            tail = num_frames - trim
            trim_len = max(tail - head, 1)
            idxs = pybvh_ml.uniform_temporal_sample(
                trim_len, self.clip_length, mode=self.mode, rng=self._rng,
            )
            idxs = (idxs % trim_len) + head
            frame_indices = idxs
        else:
            frame_indices = pybvh_ml.uniform_temporal_sample(
                num_frames, self.clip_length, mode=self.mode, rng=self._rng,
            )
            frame_indices = frame_indices % num_frames
        data_ctv = data_ctv[:, frame_indices, :]  # (C, clip_length, V)

        # 4. Joint dropout (PartAug): zero out one body-part with given probability
        if self.mode == "train" and self.joint_dropout_prob > 0.0:
            if self._rng.random() < self.joint_dropout_prob:
                part_idx = self._rng.integers(0, len(self._body_parts))
                nodes_to_drop = self._body_parts[part_idx]
                # data_ctv shape: (C, T, V)
                for n in nodes_to_drop:
                    data_ctv[:, :, n] = 0.0

        # 4b. sequence cutout per body-part (Darragh 8th place ASL Signs)
        if self.mode == "train" and self.seq_cutout_p > 0.0:
            from diema.data.augmentations import sequence_cutout_per_body_part
            data_ctv = sequence_cutout_per_body_part(
                data_ctv,
                rng=self._rng,
                body_parts=self._body_parts_dict,
                p=self.seq_cutout_p,
                num_segments=self.seq_cutout_segments,
                segment_ratio=self.seq_cutout_ratio,
            )

        # 4c. time shift delta features (lag stack)
        if self.time_shift_lags is not None:
            from diema.data.augmentations import time_shift_delta_features
            data_ctv = time_shift_delta_features(
                data_ctv, lags=self.time_shift_lags, fill_mode="edge",
            )

        # 4d. body-scale jitter — isotropic random scale on
        # "body geometry" channels. Concretely:
        #   - rotation_6d: no-op. Rotations are scale-invariant and
        #     virtual_root (CTV node 0, channels 0..2) holds *global
        #     translation*, not body size. Scaling it would be "root
        #     translation jitter" under the wrong name, so we skip the
        #     augmentation entirely for this stream.
        #   - joint_pos / bone / joint_motion / bone_motion: scale nodes
        #     1..24 only. Node 0 holds root translation in these streams
        #     too, and we want to vary body geometry, not global position.
        if (
            self.mode == "train"
            and self.body_scale_jitter_range is not None
            and self.stream_type != "rotation_6d"
        ):
            from diema.data.augmentations import body_scale_jitter
            # Skip virtual_root (CTV node 0); scale joints 1..24 only.
            pos_nodes_arg: tuple[int, ...] = tuple(range(1, 25))
            data_ctv = body_scale_jitter(
                data_ctv,
                rng=self._rng,
                scale_range=self.body_scale_jitter_range,
                pos_channels=(0, 1, 2),
                pos_nodes=pos_nodes_arg,
            )

        # 5. Return
        label = int(self.labels[idx]) if self.labels is not None else -1
        filename = self.filenames[idx] if self.filenames else ""

        return (
            torch.tensor(data_ctv, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            filename,
        )
