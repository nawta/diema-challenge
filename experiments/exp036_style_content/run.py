"""exp036: Content/Style Dual-Head on Region-Aware backbone.

Wraps the region_aware_conv1d_transformer backbone in a
:class:`diema.models.dual_head.DualHead_Model` and trains with
:class:`diema.training.dual_head_trainer.DualHeadLightningModel`. The
joint loss is::

    total = CE(emo_logits, y_emotion)
          + loss_weight_style  * CE(style_logits, y_performer)
          + loss_weight_adv    * CE(adv_logits,   y_performer)   # GRL applied in model
          + loss_weight_ortho  * orthogonality_loss(z_content, z_style)

Performer labels are derived from each sample's filename via a
``filename → performer_index`` map built from ``train_data.csv``.

Related: diema/models/dual_head/dual_head_model.py,
         diema/training/dual_head_trainer.py
         experiments/exp034_regionaware_convtr/run.py (backbone reference)

Usage:
    python -m experiments.exp036_style_content.run fold=0
    python -m experiments.exp036_style_content.run fold=0 max_epochs=5
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, CSVLogger

import pybvh_ml

from utils.env import EnvConfig
from utils.seed import seed_everything
from utils.logger import get_logger
from diema.models.skeleton_graph import DIEMA_INWARD_EDGES, DIEMA_LR_PAIRS_BVH

WANDB_PROJECT_NAME = "diema-challenge"


@dataclass
class ExpConfig:
    debug: bool = False
    seed: int = 42
    fold: int = 0
    num_folds: int = 10
    exp_tag: str = "a00"

    # data
    data_path: str = ""
    train_csv_path: str = "data/diema_challenge/raw/train_data.csv"
    target_repr: str = "6d"
    clip_length: int = 64
    num_workers: int = 8

    # backbone
    model_name: str = "region_aware_conv1d_transformer"
    num_class: int = 12
    in_channels: int = 6
    dim: int = 160
    num_conv_blocks: int = 3
    kernel_size: int = 7
    num_cross_blocks: int = 2
    num_heads: int = 4
    mlp_ratio: float = 2.0
    drop_rate: float = 0.3
    late_dropout: float = 0.8
    late_dropout_start_step: int = 1000
    use_per_part_gate: bool = True

    # dual-head
    content_dim: int = 256
    style_dim: int = 256
    proj_dropout: float = 0.1
    grl_lambda: float = 0.05
    loss_weight_style: float = 1.0
    loss_weight_adv: float = 1.0
    loss_weight_ortho: float = 0.01

    # skeleton
    num_nodes: int = 25
    inward_edges: list = field(default_factory=lambda: list(DIEMA_INWARD_EDGES))
    lr_joint_pairs: list = field(default_factory=lambda: list(DIEMA_LR_PAIRS_BVH))

    # training
    batch_size: int = 128
    max_epochs: int = 80
    optimizer: str = "AdamW"
    lr: float = 4.0e-3
    weight_decay: float = 5.0e-4
    scheduler_type: str = "cosine_warmup"
    warmup_epochs: int = 5
    early_stopping_patience: int = 15
    loss_type: str = "ce"
    label_smoothing: float = 0.3

    # augmentation (baseline; aug disabled for cleanliness)
    augment_rotate: bool = True
    augment_mirror: bool = True
    augment_speed: bool = True
    augment_noise_sigma: float = 2.5


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    exp_cfg = OmegaConf.structured(ExpConfig)
    known_keys = set(OmegaConf.to_container(exp_cfg).keys())
    filtered = {
        k: v
        for k, v in OmegaConf.to_container(cfg, resolve=True).items()
        if k in known_keys
    }
    cfg = OmegaConf.merge(exp_cfg, OmegaConf.create(filtered))

    env = EnvConfig()
    exp_name = f"exp036_style_content_{cfg.exp_tag}/fold_{cfg.fold:02d}"
    output_dir = env.artifacts_dir / exp_name

    LOGGER = get_logger(__name__, output_dir)
    seed_everything(cfg.seed)
    LOGGER.info(f"Experiment: {exp_name}")
    LOGGER.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # Data
    if cfg.data_path:
        data_path = Path(cfg.data_path)
        if not data_path.is_absolute():
            data_path = env.project_root / data_path
    else:
        data_path = env.processed_dir / "motion_train_quat.npz"
    if not data_path.exists():
        LOGGER.error(f"Data not found: {data_path}")
        return

    preprocessed = pybvh_ml.load_preprocessed(str(data_path))
    filenames = preprocessed["filenames"]

    from diema.data.splits import generate_lpo_splits
    splits = generate_lpo_splits(filenames, num_folds=cfg.num_folds)
    split_dict = splits[cfg.fold]
    LOGGER.info(
        f"Fold {cfg.fold}: train={len(split_dict['train'])}, val={len(split_dict['val'])}"
    )

    # Augmentation pipeline
    aug_steps = []
    if cfg.augment_rotate:
        aug_steps.append((
            pybvh_ml.rotate_quaternions_vertical, 1.0,
            {"angle_deg": lambda rng: rng.uniform(-180, 180), "up_idx": 2},
        ))
    if cfg.augment_mirror:
        aug_steps.append((
            pybvh_ml.mirror_quaternions, 0.5,
            {"lr_joint_pairs": list(cfg.lr_joint_pairs), "lateral_idx": 0},
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

    # DataModule
    from diema.data.collate import MotionDataModule
    datamodule = MotionDataModule(
        data_path=str(data_path),
        split_dict=split_dict,
        clip_length=cfg.clip_length,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        target_repr=cfg.target_repr,
        augmentation_pipeline=aug_pipeline,
        debug=cfg.debug,
    )

    # Performer map from train_data.csv
    from diema.training.dual_head_trainer import build_filename_to_performer_index
    filename_to_perf_idx, num_performer = build_filename_to_performer_index(
        cfg.train_csv_path
    )
    LOGGER.info(
        f"Loaded {len(filename_to_perf_idx)} filename->perf entries "
        f"({num_performer} unique performers)"
    )

    # Build backbone (Region-Aware Conv1D+Tr)
    from types import SimpleNamespace
    from diema.models import build_model
    backbone = build_model(SimpleNamespace(
        model=SimpleNamespace(
            name=cfg.model_name,
            num_class=cfg.num_class,
            in_channels=cfg.in_channels,
            clip_length=cfg.clip_length,
            dim=cfg.dim,
            num_conv_blocks=cfg.num_conv_blocks,
            kernel_size=cfg.kernel_size,
            num_cross_blocks=cfg.num_cross_blocks,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            drop_rate=cfg.drop_rate,
            late_dropout=cfg.late_dropout,
            late_dropout_start_step=cfg.late_dropout_start_step,
            use_per_part_gate=cfg.use_per_part_gate,
        ),
        skeleton=SimpleNamespace(num_nodes=cfg.num_nodes, inward_edges=list(cfg.inward_edges)),
    ))

    # Wrap backbone in DualHead_Model
    from diema.models.dual_head import DualHead_Model
    model = DualHead_Model(
        backbone=backbone,
        feature_dim=backbone.feature_dim,
        num_class=cfg.num_class,
        num_performer=num_performer,
        content_dim=cfg.content_dim,
        style_dim=cfg.style_dim,
        proj_dropout=cfg.proj_dropout,
        grl_lambda=cfg.grl_lambda,
    )

    n_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"DualHead model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    loss_kwargs = {}
    if cfg.label_smoothing > 0:
        loss_kwargs["label_smoothing"] = cfg.label_smoothing

    # DualHead Lightning wrapper
    from diema.training.dual_head_trainer import DualHeadLightningModel
    lit_model = DualHeadLightningModel(
        model=model,
        base_lr=cfg.lr,
        num_class=cfg.num_class,
        filename_to_perf_idx=filename_to_perf_idx,
        num_performer=num_performer,
        loss_weight_style=cfg.loss_weight_style,
        loss_weight_adv=cfg.loss_weight_adv,
        loss_weight_ortho=cfg.loss_weight_ortho,
        loss_type=cfg.loss_type,
        loss_kwargs=loss_kwargs,
        optimizer=cfg.optimizer,
        scheduler_type=cfg.scheduler_type,
        weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs,
    )

    # Callbacks
    from diema.training.callbacks import (
        MetricsSaveCallback, ConfigSaveCallback, OOFPredictionCallback,
    )
    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir), filename="best",
            monitor="val_f1", mode="max", save_top_k=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=cfg.early_stopping_patience, mode="min",
        ),
        MetricsSaveCallback(output_dir=output_dir),
        ConfigSaveCallback(config=OmegaConf.to_container(cfg, resolve=True), output_dir=output_dir),
        OOFPredictionCallback(output_dir=output_dir, monitor="val_f1", mode="max"),
    ]

    loggers = [CSVLogger(save_dir=str(output_dir), name="csv_logs")]
    try:
        loggers.append(WandbLogger(
            project=WANDB_PROJECT_NAME, name=exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        ))
    except Exception:
        LOGGER.warning("wandb not available, using CSV logger only")

    torch.set_float32_matmul_precision("high")
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        precision=32,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
        deterministic=True,
    )

    LOGGER.info("Starting training...")
    trainer.fit(lit_model, datamodule=datamodule)

    LOGGER.info("Running test evaluation...")
    # Use Lightning's "best" sentinel so we reload *this* run's best
    # checkpoint (tracked by ModelCheckpoint), not a stale best.ckpt
    # left behind by a previous smoke run in the same output dir.
    trainer.test(lit_model, datamodule=datamodule, ckpt_path="best")

    LOGGER.info(f"Done. Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
