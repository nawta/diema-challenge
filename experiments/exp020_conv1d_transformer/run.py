"""exp020: Conv1D + Transformer hybrid (ASL Signs 1st place / hoyso48).

Faithful PyTorch port of the 1.85M-param 1DCNN+Transformer solution that
won Kaggle's ASL Signs competition (LB 0.80 single-model). The model is
registered as `"conv1d_transformer"` and produces diema's standard
BaseModel contract: input (N, 6, 64, 25) -> {"logits": (N, 12)}.

Default settings (a00):
  - clip=64, batch=128, AdamW lr=4e-3, cosine_warmup (warmup 5ep / 80ep total)
  - label_smoothing=0.1
  - augmentation: rotate / mirror / speed / noise (same as exp003)

Related: diema/models/conv1d_transformer/conv1d_transformer_model.py

Usage:
    python -m experiments.exp020_conv1d_transformer.run fold=0
    python -m experiments.exp020_conv1d_transformer.run fold=0 exp_tag=a01
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
    # Phase ACC Track A: stream_type wiring (bone/bone_motion members). See ~/.claude/plans/replicated-gathering-raven.md
    data_path: str = ""
    target_repr: str = "6d"
    stream_type: str = "rotation_6d"
    clip_length: int = 64
    num_workers: int = 8

    # model
    model_name: str = "conv1d_transformer"
    num_class: int = 12
    in_channels: int = 6
    dim: int = 192
    num_blocks: int = 2
    kernel_size: int = 17
    num_heads: int = 4
    mlp_ratio: float = 2.0
    drop_rate: float = 0.2
    late_dropout: float = 0.8
    late_dropout_start_step: int = 1000
    pool_type: str = "gap"

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
    label_smoothing: float = 0.1

    # augmentation (same as exp003 baseline)
    augment_rotate: bool = True
    augment_mirror: bool = True
    augment_speed: bool = True
    augment_noise_sigma: float = 2.5

    # domain generalization package (per-sample, opt-in)
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
    exp_name = f"exp020_conv1d_transformer_{cfg.exp_tag}/fold_{cfg.fold:02d}"
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

    # Augmentation
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
        stream_type=cfg.stream_type,
        augmentation_pipeline=aug_pipeline,
        debug=cfg.debug,
        seq_cutout_p=cfg.seq_cutout_p,
        seq_cutout_segments=cfg.seq_cutout_segments,
        seq_cutout_ratio=cfg.seq_cutout_ratio,
        body_scale_jitter_range=body_scale_range,
    )

    # Fail-fast: verify in_channels matches the chosen stream (prevents silent misconfig).
    expected_ch = MotionDataModule.expected_in_channels(cfg.stream_type, None)
    if cfg.in_channels != expected_ch:
        raise ValueError(
            f"in_channels mismatch: cfg.in_channels={cfg.in_channels} but "
            f"stream_type='{cfg.stream_type}' expects {expected_ch} channels. "
            f"Pass in_channels={expected_ch} when using stream_type='{cfg.stream_type}'."
        )
    LOGGER.info(f"stream_type='{cfg.stream_type}', in_channels={cfg.in_channels} (OK)")

    # Model
    from types import SimpleNamespace
    from diema.models import build_model
    model = build_model(SimpleNamespace(
        model=SimpleNamespace(
            name=cfg.model_name, num_class=cfg.num_class, in_channels=cfg.in_channels,
            dim=cfg.dim, num_blocks=cfg.num_blocks, kernel_size=cfg.kernel_size,
            num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio, drop_rate=cfg.drop_rate,
            late_dropout=cfg.late_dropout, late_dropout_start_step=cfg.late_dropout_start_step,
            pool_type=cfg.pool_type,
        ),
        skeleton=SimpleNamespace(num_nodes=cfg.num_nodes, inward_edges=list(cfg.inward_edges)),
    ))

    loss_kwargs = {}
    if cfg.label_smoothing > 0:
        loss_kwargs["label_smoothing"] = cfg.label_smoothing

    from diema.training.train import LightningModel
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
        mixup_alpha=cfg.mixup_alpha,
        cutmix_alpha=cfg.cutmix_alpha,
    )

    # Callbacks: monitor val_f1 (primary metric per)
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

    LOGGER.info("Starting training...")
    trainer.fit(lit_model, datamodule=datamodule)

    LOGGER.info("Running test evaluation...")
    # Use Lightning's "best" sentinel so we reload *this* run's best
    # checkpoint (tracked by the ModelCheckpoint callback), not a stale
    # best.ckpt left behind by a previous smoke run in the same output dir.
    trainer.test(lit_model, datamodule=datamodule, ckpt_path="best")

    LOGGER.info(f"Done. Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
