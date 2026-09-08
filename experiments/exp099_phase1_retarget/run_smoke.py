"""run_smoke.py — Stage 3 of exp099

Train SkateFormer in one of three smoke modes, write metrics + best
checkpoint + OOF preds. Designed to be called per (mode, fold) by
`train_smoke.sh`.

Modes
-----
- `rotation_6d`:   production-replica baseline (no retarget, in_channels=6).
                   Uses diema.data.collate.MotionDataModule.
- `jointpos_honest`: joint_pos input (in_channels=3) + honest per-clip FK
                   on originals + retargeted mix (mix_ratio=0.4).
- `jointpos_flat`: joint_pos input (in_channels=3) + JP_06-flattened FK
                   on originals + retargeted mix (mix_ratio=0.4).

Hyperparameters fixed at production a01 (5e-4 AdamW + cosine_warmup +
batch_size=128 + label_smoothing=0.1), with max_epochs reduced to 20
for smoke (vs 65 production).

Usage:
    python experiments/exp099_phase1_retarget/run_smoke.py \\
        mode=rotation_6d fold=0

    python experiments/exp099_phase1_retarget/run_smoke.py \\
        mode=jointpos_honest fold=0 mix_ratio=0.4
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from omegaconf import OmegaConf

import pybvh_ml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "exp099_phase1_retarget"))

from utils.env import EnvConfig
from utils.seed import seed_everything
from utils.logger import get_logger
from diema.models.skeleton_graph import DIEMA_INWARD_EDGES, DIEMA_LR_PAIRS_BVH
from diema.data.splits import generate_lpo_splits
from diema.data.collate import MotionDataModule
from diema.models import build_model
from diema.training.train import LightningModel
from diema.training.callbacks import (
    MetricsSaveCallback, ConfigSaveCallback, OOFPredictionCallback,
)

from mixed_datamodule import MixedJointPosDataModule


WANDB_PROJECT = "acii2026-retarget-phase1-smoke"


@dataclass
class SmokeConfig:
    # Mode + fold
    mode: str = "rotation_6d"   # rotation_6d / jointpos_honest / jointpos_flat
    fold: int = 0
    num_folds: int = 10
    seed: int = 42

    # Data
    data_path: str = ""
    clip_length: int = 64
    num_workers: int = 8
    mix_ratio: float = 0.4

    # Model (SkateFormer; in_channels is set by mode)
    num_class: int = 12
    dim: int = 128
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    window_size: int = 16
    dropout: float = 0.3
    drop_path: float = 0.1
    pool_type: str = "gap"
    num_nodes: int = 25

    # Training (matches production a01)
    batch_size: int = 128
    max_epochs: int = 20       # smoke
    optimizer: str = "AdamW"
    lr: float = 5.0e-4
    weight_decay: float = 0.05
    scheduler_type: str = "cosine_warmup"
    warmup_epochs: int = 5
    early_stopping_patience: int = 8
    loss_type: str = "ce"
    label_smoothing: float = 0.1

    # Augmentation
    augment_rotate: bool = True
    augment_mirror: bool = True
    augment_speed: bool = True
    augment_noise_sigma: float = 2.5

    # wandb
    wandb_project: str = WANDB_PROJECT

    # Dry-run: shorten training to a handful of batches/epoch for pipeline sanity
    dry_run: bool = False


def _parse_cli() -> SmokeConfig:
    """Tiny hand-rolled key=value CLI to avoid Hydra path mismatch."""
    cfg = OmegaConf.structured(SmokeConfig())
    overrides = {}
    for arg in sys.argv[1:]:
        if "=" not in arg:
            raise ValueError(f"Bad arg (need key=val): {arg!r}")
        k, v = arg.split("=", 1)
        overrides[k] = v
    cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return OmegaConf.to_object(cfg)


def _build_skateformer(cfg: SmokeConfig, in_channels: int):
    from types import SimpleNamespace
    return build_model(SimpleNamespace(
        model=SimpleNamespace(
            name="skateformer",
            num_class=cfg.num_class,
            in_channels=in_channels,
            dim=cfg.dim, depth=cfg.depth, num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio, window_size=cfg.window_size,
            dropout=cfg.dropout,
            attn_drop=0.0, proj_drop=0.0,
            drop_path=cfg.drop_path,
            pool_type=cfg.pool_type, use_arcface_head=False,
        ),
        skeleton=SimpleNamespace(
            num_nodes=cfg.num_nodes, inward_edges=list(DIEMA_INWARD_EDGES)),
    ))


def main():
    cfg = _parse_cli()
    env = EnvConfig()
    exp_name = f"exp099_phase1_smoke/{cfg.mode}_fold_{cfg.fold:02d}"
    output_dir = env.artifacts_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER = get_logger(__name__, output_dir)
    seed_everything(cfg.seed)
    LOGGER.info(f"Experiment: {exp_name}")
    LOGGER.info(f"Mode: {cfg.mode}, fold={cfg.fold}, mix_ratio={cfg.mix_ratio}")

    data_path = (Path(cfg.data_path) if cfg.data_path
                  else env.processed_dir / "motion_train_quat.npz")
    if not data_path.exists():
        LOGGER.error(f"Data not found: {data_path}")
        sys.exit(2)

    preprocessed = pybvh_ml.load_preprocessed(str(data_path))
    filenames = preprocessed["filenames"]
    splits = generate_lpo_splits(filenames, num_folds=cfg.num_folds)
    split_dict = splits[cfg.fold]
    LOGGER.info(f"Fold {cfg.fold}: train={len(split_dict['train'])}, "
                f"val={len(split_dict['val'])}")

    # ------------------------------------------------------------------
    # DataModule + model — mode-dispatched
    # ------------------------------------------------------------------
    if cfg.mode == "rotation_6d":
        in_channels = 6
        # Build pybvh_ml augmentation pipeline (same as production a01)
        aug_steps = []
        if cfg.augment_rotate:
            # up_idx=2 (Z) — DIEM-A skeleton is Z-up in world space; a true
            # yaw rotation must rotate around axis 2. Production exp004 used
            # up_idx=1 (Y), which tumbles the figure. This smoke uses the
            # corrected axis for BOTH modes (fair + correct comparison).
            aug_steps.append((
                pybvh_ml.rotate_quaternions_vertical, 1.0,
                {"angle_deg": lambda rng: rng.uniform(-180, 180), "up_idx": 2},
            ))
        if cfg.augment_mirror:
            aug_steps.append((
                pybvh_ml.mirror_quaternions, 0.5,
                {"lr_joint_pairs": list(DIEMA_LR_PAIRS_BVH), "lateral_idx": 0},
            ))
        if cfg.augment_speed:
            aug_steps.append((
                pybvh_ml.speed_perturbation_arrays, 1.0,
                {"factor": lambda rng: rng.uniform(0.8, 1.2)},
            ))
        if cfg.augment_noise_sigma > 0:
            aug_steps.append((
                pybvh_ml.add_joint_noise_quaternions, 1.0,
                {"sigma_deg": cfg.augment_noise_sigma},
            ))
        aug_pipeline = pybvh_ml.AugmentationPipeline(aug_steps) if aug_steps else None
        datamodule = MotionDataModule(
            data_path=str(data_path),
            split_dict=split_dict,
            clip_length=cfg.clip_length,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            target_repr="6d",
            augmentation_pipeline=aug_pipeline,
            seed=cfg.seed,
            stream_type="rotation_6d",
            sampling_mode="uniform",
        )
    elif cfg.mode in ("jointpos_honest", "jointpos_flat"):
        in_channels = 3
        fk_rule = "honest" if cfg.mode == "jointpos_honest" else "flat"
        datamodule = MixedJointPosDataModule(
            split_dict=split_dict,
            cached_npz_path=str(data_path),
            original_fk_rule=fk_rule,
            mix_ratio=cfg.mix_ratio,
            clip_length=cfg.clip_length,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            augment_rotate=cfg.augment_rotate,
            augment_mirror=cfg.augment_mirror,
            augment_speed=cfg.augment_speed,
            augment_noise_sigma=cfg.augment_noise_sigma,
            lr_joint_pairs=list(DIEMA_LR_PAIRS_BVH),
            seed=cfg.seed,
        )
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    # Set up to surface dataset summary (joint_pos modes)
    datamodule.setup("fit")
    if hasattr(datamodule, "dataset_train") and hasattr(
            datamodule.dataset_train, "summary"):
        LOGGER.info(f"Train dataset summary: "
                    f"{datamodule.dataset_train.summary()}")

    model = _build_skateformer(cfg, in_channels=in_channels)

    loss_kwargs = {}
    if cfg.label_smoothing > 0:
        loss_kwargs["label_smoothing"] = cfg.label_smoothing

    lit_model = LightningModel(
        model=model,
        base_lr=cfg.lr,
        num_class=cfg.num_class,
        loss_type=cfg.loss_type,
        loss_kwargs=loss_kwargs,
        optimizer=cfg.optimizer,
        scheduler_type=cfg.scheduler_type,
        weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs,
    )

    callbacks = [
        ModelCheckpoint(dirpath=str(output_dir), filename="best",
                        monitor="val_f1", mode="max", save_top_k=1),
        EarlyStopping(monitor="val_loss",
                      patience=cfg.early_stopping_patience, mode="min"),
        MetricsSaveCallback(output_dir=output_dir),
        ConfigSaveCallback(
            config={k: v for k, v in cfg.__dict__.items()},
            output_dir=output_dir),
        OOFPredictionCallback(output_dir=output_dir, monitor="val_f1", mode="max"),
    ]

    loggers = [CSVLogger(save_dir=str(output_dir), name="csv_logs")]
    try:
        loggers.append(WandbLogger(
            project=cfg.wandb_project,
            name=exp_name.replace("/", "__"),
            config={k: v for k, v in cfg.__dict__.items()},
        ))
    except Exception as e:
        LOGGER.warning(f"wandb not available: {e}")

    torch.set_float32_matmul_precision("high")
    trainer_kwargs = dict(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        precision=32,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
        deterministic=True,
    )
    if cfg.dry_run:
        trainer_kwargs.update(dict(
            max_epochs=1,
            limit_train_batches=5,
            limit_val_batches=5,
            limit_test_batches=5,
        ))
        LOGGER.info("DRY RUN: 1 epoch, 5 batches each split")
    trainer = pl.Trainer(**trainer_kwargs)

    LOGGER.info("Starting training...")
    trainer.fit(lit_model, datamodule=datamodule)
    LOGGER.info("Running test evaluation...")
    trainer.test(lit_model, datamodule=datamodule, ckpt_path="best")
    LOGGER.info(f"Done. Artifacts → {output_dir}")


if __name__ == "__main__":
    main()
