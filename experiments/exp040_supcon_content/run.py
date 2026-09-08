"""exp040: Performer-Aware SupCon on Content branch.

Uses the Conv1D+Transformer backbone (exp020 a01 settings) with a
SupCon projection head. The combined loss is CE + λ·SupCon, where
positives from *different* performers receive higher weight to
encourage cross-performer content alignment.

Default settings (a00):
  - backbone: conv1d_transformer, dim=192, 2 blocks, k=17 (exp020 a01)
  - proj_dim=128, supcon_lambda=0.05, warmup 5 ep CE-only
  - supcon_temperature=0.1, w_strong=2.0, w_weak=0.5
  - label_smoothing=0.3, drop_rate=0.3 (exp020 a01 best settings)

Related: diema/models/conv1d_transformer/supcon_conv1d_tr_model.py,
         diema/training/supcon_trainer.py
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
    target_repr: str = "6d"
    clip_length: int = 64
    num_workers: int = 8

    # backbone (matches exp020 a01 best settings)
    model_name: str = "conv1d_transformer"
    num_class: int = 12
    in_channels: int = 6
    dim: int = 192
    num_blocks: int = 2
    kernel_size: int = 17
    num_heads: int = 4
    mlp_ratio: float = 2.0
    drop_rate: float = 0.3
    late_dropout: float = 0.8
    late_dropout_start_step: int = 1000
    pool_type: str = "gap"

    # SupCon head
    proj_dim: int = 128

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

    # SupCon loss (a00)
    supcon_lambda: float = 0.05
    supcon_warmup_epochs: int = 5
    supcon_temperature: float = 0.1
    supcon_w_strong: float = 2.0
    supcon_w_weak: float = 0.5

    # CenterLoss (a01, SupCon replacement for small data)
    use_center_loss: bool = False
    lambda_center: float = 0.001
    lambda_tau: float = 20.0
    center_lr: float = 0.5

    # augmentation
    augment_rotate: bool = True
    augment_mirror: bool = True
    augment_speed: bool = True
    augment_noise_sigma: float = 2.5

    # aug (opt-in)
    seq_cutout_p: float = 0.0
    seq_cutout_segments: int = 5
    seq_cutout_ratio: float = 0.15
    body_scale_jitter_lo: float = 0.0
    body_scale_jitter_hi: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    exp_cfg = OmegaConf.structured(ExpConfig)
    known_keys = set(OmegaConf.to_container(exp_cfg).keys())
    filtered = {k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items()
                if k in known_keys}
    cfg = OmegaConf.merge(exp_cfg, OmegaConf.create(filtered))

    env = EnvConfig()
    exp_name = f"exp040_supcon_content_{cfg.exp_tag}/fold_{cfg.fold:02d}"
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
    LOGGER.info(f"Fold {cfg.fold}: train={len(split_dict['train'])}, val={len(split_dict['val'])}")

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
    body_scale_range = None
    if cfg.body_scale_jitter_lo > 0 and cfg.body_scale_jitter_hi > 0:
        body_scale_range = (float(cfg.body_scale_jitter_lo), float(cfg.body_scale_jitter_hi))
    datamodule = MotionDataModule(
        data_path=str(data_path),
        split_dict=split_dict,
        clip_length=cfg.clip_length,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        target_repr=cfg.target_repr,
        augmentation_pipeline=aug_pipeline,
        debug=cfg.debug,
        seq_cutout_p=cfg.seq_cutout_p,
        seq_cutout_segments=cfg.seq_cutout_segments,
        seq_cutout_ratio=cfg.seq_cutout_ratio,
        body_scale_jitter_range=body_scale_range,
    )

    # Model
    from types import SimpleNamespace
    from diema.models import build_model

    backbone_cfg = SimpleNamespace(
        model=SimpleNamespace(
            name=cfg.model_name, num_class=cfg.num_class, in_channels=cfg.in_channels,
            dim=cfg.dim, num_blocks=cfg.num_blocks, kernel_size=cfg.kernel_size,
            num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio, drop_rate=cfg.drop_rate,
            late_dropout=cfg.late_dropout, late_dropout_start_step=cfg.late_dropout_start_step,
            pool_type=cfg.pool_type,
        ),
        skeleton=SimpleNamespace(num_nodes=cfg.num_nodes, inward_edges=list(cfg.inward_edges)),
    )
    backbone = build_model(backbone_cfg)

    loss_kwargs = {}
    if cfg.label_smoothing > 0:
        loss_kwargs["label_smoothing"] = cfg.label_smoothing

    if cfg.use_center_loss:
        # a01: CenterLoss mode — use backbone directly (no proj head)
        model = backbone
        n_params = sum(p.numel() for p in model.parameters())
        LOGGER.info(f"Model parameters: {n_params:,} (CenterLoss mode, no proj head)")

        from diema.training.center_loss_trainer import CenterLossLightningModel
        lit_model = CenterLossLightningModel(
            model=model,
            base_lr=cfg.lr,
            num_class=cfg.num_class,
            feat_dim=backbone.feature_dim,
            lambda_center=cfg.lambda_center,
            lambda_tau=cfg.lambda_tau,
            center_lr=cfg.center_lr,
            loss_type=cfg.loss_type,
            loss_kwargs=loss_kwargs,
            optimizer=cfg.optimizer,
            scheduler_type=cfg.scheduler_type,
            weight_decay=cfg.weight_decay,
            warmup_epochs=cfg.warmup_epochs,
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
        )
    else:
        # a00: SupCon mode — backbone + projection head
        from diema.models.conv1d_transformer.supcon_conv1d_tr_model import SupConConv1DTr_Model
        model = SupConConv1DTr_Model(backbone=backbone, proj_dim=cfg.proj_dim)
        n_params = sum(p.numel() for p in model.parameters())
        LOGGER.info(f"Model parameters: {n_params:,} (SupCon mode)")

        from diema.training.dual_head_trainer import (
            build_filename_to_performer_index,
            build_pair_to_performer_index,
        )
        train_csv = Path("data/diema_challenge/raw/train_data.csv")
        if train_csv.exists():
            fn_map, num_perf = build_filename_to_performer_index(train_csv)
            pair_map, _ = build_pair_to_performer_index(train_csv)
            LOGGER.info(f"Performer mapping: {num_perf} performers, {len(fn_map)} filenames")
        else:
            LOGGER.warning(f"train_data.csv not found, disabling performer-aware SupCon")
            fn_map, num_perf, pair_map = {}, 0, {}

        from diema.training.supcon_trainer import SupConLightningModel
        lit_model = SupConLightningModel(
            model=model,
            base_lr=cfg.lr,
            num_class=cfg.num_class,
            filename_to_perf_idx=fn_map,
            num_performer=num_perf,
            pair_to_perf_idx=pair_map,
            supcon_lambda=cfg.supcon_lambda,
            supcon_warmup_epochs=cfg.supcon_warmup_epochs,
            supcon_temperature=cfg.supcon_temperature,
            supcon_w_strong=cfg.supcon_w_strong,
            supcon_w_weak=cfg.supcon_w_weak,
            loss_type=cfg.loss_type,
            loss_kwargs=loss_kwargs,
            optimizer=cfg.optimizer,
            scheduler_type=cfg.scheduler_type,
            weight_decay=cfg.weight_decay,
            warmup_epochs=cfg.warmup_epochs,
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
        )

    # Callbacks
    from diema.training.callbacks import MetricsSaveCallback, ConfigSaveCallback, OOFPredictionCallback
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

    trainer.fit(lit_model, datamodule=datamodule)
    trainer.test(lit_model, datamodule=datamodule, ckpt_path="best")

    best_f1 = trainer.callback_metrics.get("val_f1", 0)
    best_acc = trainer.callback_metrics.get("val_acc", 0)
    LOGGER.info(f"Best val_f1: {best_f1:.4f}, val_acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
