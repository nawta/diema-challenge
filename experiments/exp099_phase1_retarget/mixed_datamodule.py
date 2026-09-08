"""mixed_datamodule.py — LightningDataModule for JointPosMixedDataset.

Thin wrapper so the existing pl.Trainer + LightningModel + callbacks
pipeline can be reused. Mirrors `diema.data.collate.MotionDataModule`
but instantiates `JointPosMixedDataset` instead.

Related:
- experiments/exp099_phase1_retarget/mixed_dataset.py (JointPosMixedDataset)
- diema/data/collate.py (MotionDataModule, motion_collate_fn)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytorch_lightning as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "exp099_phase1_retarget"))

from diema.data.collate import motion_collate_fn, _worker_init_fn, _default_num_workers  # noqa: E402
from diema.data.parser import parse_performer_id  # noqa: E402
from mixed_dataset import JointPosMixedDataset, _build_augmentation_pipeline  # noqa: E402


class MixedJointPosDataModule(pl.LightningDataModule):
    """LightningDataModule wrapping JointPosMixedDataset.

    Mix ratio / FK rule are applied only to the training split. Val/test
    splits get original-only joint_pos under the same FK rule (so the
    metric reflects "would this model generalize to held-out performers
    on real DIEM-A data?", not augmented data).

    Parameters
    ----------
    split_dict : dict with keys 'train', 'val' (each a list of (fname, idx) tuples)
    original_fk_rule : "honest" | "flat"
    mix_ratio : float
    clip_length : int
    batch_size : int
    num_workers : int | None
    augment_rotate / augment_mirror / augment_speed / augment_noise_sigma : training augmentations
    lr_joint_pairs : list of (left, right) joint index pairs (for mirror aug)
    seed : int
    """

    def __init__(
        self,
        split_dict: dict,
        *,
        cached_npz_path: str | None = None,
        original_fk_rule: str = "honest",
        mix_ratio: float = 0.4,
        clip_length: int = 64,
        batch_size: int = 128,
        num_workers: int | None = None,
        augment_rotate: bool = True,
        augment_mirror: bool = True,
        augment_speed: bool = True,
        augment_noise_sigma: float = 2.5,
        lr_joint_pairs: list | None = None,
        seed: int = 42,
    ):
        super().__init__()
        if lr_joint_pairs is None:
            from diema.models.skeleton_graph import DIEMA_LR_PAIRS_BVH
            lr_joint_pairs = list(DIEMA_LR_PAIRS_BVH)
        from mixed_dataset import CACHED_NPZ_DEFAULT
        self.split_dict = split_dict
        self.cached_npz_path = cached_npz_path or CACHED_NPZ_DEFAULT
        self.original_fk_rule = original_fk_rule
        self.mix_ratio = mix_ratio
        self.clip_length = clip_length
        self.batch_size = batch_size
        self.num_workers = (
            num_workers if num_workers is not None else _default_num_workers()
        )
        self.augment_rotate = augment_rotate
        self.augment_mirror = augment_mirror
        self.augment_speed = augment_speed
        self.augment_noise_sigma = augment_noise_sigma
        self.lr_joint_pairs = lr_joint_pairs
        self.seed = seed

    def setup(self, stage: str):
        # Performer set in train fold (used to filter retargeted pool).
        train_performers = {
            parse_performer_id(fn) for fn, _ in self.split_dict["train"]
        }

        aug_pipe = _build_augmentation_pipeline(
            augment_rotate=self.augment_rotate,
            augment_mirror=self.augment_mirror,
            augment_speed=self.augment_speed,
            augment_noise_sigma=self.augment_noise_sigma,
            lr_joint_pairs=self.lr_joint_pairs,
        )

        if stage == "fit":
            self.dataset_train = JointPosMixedDataset(
                original_split_entries=self.split_dict["train"],
                train_performer_ids=train_performers,
                cached_npz_path=self.cached_npz_path,
                original_fk_rule=self.original_fk_rule,
                mix_ratio=self.mix_ratio,
                clip_length=self.clip_length,
                is_test=False,
                augmentation_pipeline=aug_pipe,
                seed=self.seed,
            )
            self.dataset_val = JointPosMixedDataset(
                original_split_entries=self.split_dict["val"],
                train_performer_ids=train_performers,
                cached_npz_path=self.cached_npz_path,
                original_fk_rule=self.original_fk_rule,
                mix_ratio=0.0,    # val never sees retargeted
                clip_length=self.clip_length,
                is_test=True,
                augmentation_pipeline=None,
                seed=self.seed,
            )
        if stage in ("test", "predict"):
            self.dataset_test = JointPosMixedDataset(
                original_split_entries=self.split_dict.get(
                    "test", self.split_dict["val"]),
                train_performer_ids=train_performers,
                cached_npz_path=self.cached_npz_path,
                original_fk_rule=self.original_fk_rule,
                mix_ratio=0.0,
                clip_length=self.clip_length,
                is_test=True,
                augmentation_pipeline=None,
                seed=self.seed,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train, batch_size=self.batch_size, shuffle=True,
            drop_last=True, num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_worker_init_fn, collate_fn=motion_collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_worker_init_fn, collate_fn=motion_collate_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_worker_init_fn, collate_fn=motion_collate_fn,
        )
