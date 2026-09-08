"""Custom collate function and DataModule for motion capture data.

See: diema_challenge_implementation_spec.md §6.3 — データローダ
Related: diema/data/dataset.py (MotionDataset)

Ported from: an internal baseline (Loader class) with changes:
  - Hydra config compatible
  - Explicit augmentation pipeline construction
  - Enhanced worker init
"""

import os
import pickle
from pathlib import Path

import torch.utils.data
import pytorch_lightning as pl

from diema.data.dataset import MotionDataset


def _default_num_workers() -> int:
    """Half the available CPU cores, capped at 8."""
    return min((os.cpu_count() or 4) // 2, 8)


def _worker_init_fn(worker_id: int) -> None:
    """Re-seed each DataLoader worker's augmentation RNG independently."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if hasattr(dataset, "reseed_rng"):
            dataset.reseed_rng(worker_id)


def motion_collate_fn(batch):
    """Default collate for MotionDataset output.

    Each sample is (data_tensor, label, filename).
    Returns (batched_data, batched_labels, list_of_filenames).
    """
    data_list, label_list, filename_list = zip(*batch)
    data = torch.stack(data_list, dim=0)
    labels = torch.stack(label_list, dim=0)
    return data, labels, list(filename_list)


class MotionDataModule(pl.LightningDataModule):
    """LightningDataModule wrapping MotionDataset for train/val/test splits.

    Args:
        data_path: path to .npz data file (pybvh-ml format)
        split_path: path to split dict pickle (mutually exclusive with split_dict)
        split_dict: in-memory split dictionary (mutually exclusive with split_path)
        clip_length: number of frames to sample
        batch_size: batch size for train/val/test
        num_workers: DataLoader workers
        target_repr: target rotation representation for model input
        seed: random seed for sampling
        augmentation_pipeline: pybvh_ml.AugmentationPipeline or None
        euler_orders: per-joint Euler orders (for quat→Euler conversion)
        debug: if True, limit to 100 samples
    """

    def __init__(
        self,
        data_path: str,
        split_path: str | None = None,
        split_dict: dict | None = None,
        clip_length: int = 64,
        batch_size: int = 128,
        num_workers: int | None = None,
        target_repr: str = "6d",
        seed: int = 255,
        augmentation_pipeline=None,
        euler_orders=None,
        debug: bool = False,
        stream_type: str = "rotation_6d",
        sampling_mode: str = "uniform",
        joint_dropout_prob: float = 0.0,
        seq_cutout_p: float = 0.0,
        seq_cutout_segments: int = 5,
        seq_cutout_ratio: float = 0.15,
        time_shift_lags: tuple[int, ...] | None = None,
        body_scale_jitter_range: tuple[float, float] | None = None,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else _default_num_workers()
        self.debug = debug
        self.clip_length = clip_length
        self.target_repr = target_repr
        self.seed = seed
        self.augmentation_pipeline = augmentation_pipeline
        self.euler_orders = euler_orders
        self.stream_type = stream_type
        self.sampling_mode = sampling_mode
        self.joint_dropout_prob = joint_dropout_prob
        self.seq_cutout_p = seq_cutout_p
        self.seq_cutout_segments = seq_cutout_segments
        self.seq_cutout_ratio = seq_cutout_ratio
        self.time_shift_lags = time_shift_lags
        self.body_scale_jitter_range = body_scale_jitter_range

        if split_dict is not None:
            self.split_dict = split_dict
        elif split_path is not None:
            with open(split_path, "rb") as f:
                self.split_dict = pickle.load(f)
        else:
            raise ValueError("Either split_path or split_dict must be provided")

        self.train_indices = [idx for _, idx in self.split_dict["train"]]
        self.val_indices = [idx for _, idx in self.split_dict["val"]]

        if "test" in self.split_dict and len(self.split_dict["test"]) > 0:
            self.test_indices = [idx for _, idx in self.split_dict["test"]]
        else:
            self.test_indices = self.val_indices

        if self.debug:
            self.train_indices = self.train_indices[:100]
            self.val_indices = self.val_indices[:100]
            self.test_indices = self.test_indices[:100]

    @staticmethod
    def expected_in_channels(
        stream_type: str = "rotation_6d",
        time_shift_lags: tuple[int, ...] | None = None,
    ) -> int:
        """Return the number of input channels the model should expect.

        When ``time_shift_lags`` is set, the dataset stacks the original
        stream plus one lag block per lag, multiplying the channel count.
        Use this to keep the dataset config and model's ``in_channels``
        constructor arg in sync.

        Example:
            >>> MotionDataModule.expected_in_channels("rotation_6d", None)
            6
            >>> MotionDataModule.expected_in_channels("rotation_6d", (1, 2, 4))
            24  # 6 × (1 + 3)
        """
        from diema.features.multi_stream import STREAM_CHANNELS
        if stream_type not in STREAM_CHANNELS:
            raise ValueError(f"Unknown stream_type: {stream_type}")
        base = STREAM_CHANNELS[stream_type]
        lag_count = len(time_shift_lags) if time_shift_lags else 0
        return base * (1 + lag_count)

    def setup(self, stage: str):
        common = dict(
            data_path=str(self.data_path),
            clip_length=self.clip_length,
            target_repr=self.target_repr,
            seed=self.seed,
            euler_orders=self.euler_orders,
            stream_type=self.stream_type,
            sampling_mode=self.sampling_mode,
            joint_dropout_prob=self.joint_dropout_prob,
            seq_cutout_p=self.seq_cutout_p,
            seq_cutout_segments=self.seq_cutout_segments,
            seq_cutout_ratio=self.seq_cutout_ratio,
            time_shift_lags=self.time_shift_lags,
            body_scale_jitter_range=self.body_scale_jitter_range,
        )

        if stage == "fit":
            self.dataset_train = MotionDataset(
                indices=self.train_indices,
                is_test=False,
                augmentation_pipeline=self.augmentation_pipeline,
                **common,
            )
            self.dataset_val = MotionDataset(
                indices=self.val_indices,
                is_test=True,
                **common,
            )

        if stage in ("test", "predict"):
            self.dataset_test = MotionDataset(
                indices=self.test_indices,
                is_test=True,
                **common,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_worker_init_fn,
            collate_fn=motion_collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=motion_collate_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=motion_collate_fn,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
