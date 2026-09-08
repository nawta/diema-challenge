"""exp052: GAP-style part-wise rationale alignment.

Backbone: Region-Aware Conv1D+Transformer (exp034 a00).
Addition: PartAlignConv1DTr_Model head + CE + λ_part · PartAlignmentLoss on
Qwen3-VL rationale embeddings. Default mode is cosine (batch-local per-cell);
a02 tries InfoNCE (per-part batch contrastive).

Go judgment (TODO 3AE-6): backbone (exp034 a00) vs exp052 a00 → +0.3pt F1 on
3-fold mean. Pass → 10-fold completion + ensemble merge.

Prerequisites:
  1. sweep complete (rationale JSON for every train clip).
  2. ``tools/build_rationale_text_cache.py`` has produced
     ``output/rationale_text_cache.pt``.

Related:
- diema/models/conv1d_transformer/partalign_conv1d_tr_model.py
- diema/training/part_align_trainer.py
- diema/features/rationale_text_cache.py
- experiments/exp051_tmr_scenario_global/run.py (global scenario analogue)
- experiments/exp034_regionaware_convtr/run.py (backbone-only baseline)

Usage:
    python -m experiments.exp052_gap_generated_rationale.run fold=0
    python -m experiments.exp052_gap_generated_rationale.run fold=0 exp=a01
    python -m experiments.exp052_gap_generated_rationale.run fold=0 max_epochs=5 debug=true
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

    # backbone (exp034 a00)
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

    # part alignment head
    text_dim: int = 768
    num_parts: int = 6
    proj_hidden: int | None = None
    rationale_cache_path: str = "output/rationale_text_cache_mpnet.pt"

    # part loss
    part_loss_weight: float = 0.05
    part_loss_warmup_epochs: int = 0
    part_loss_temperature: float = 0.1
    part_loss_mode: str = "cosine"

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

    # augmentation (mirror exp034 a00)
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
    filtered = {
        k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items()
        if k in known_keys
    }
    cfg = OmegaConf.merge(exp_cfg, OmegaConf.create(filtered))

    env = EnvConfig()
    exp_name = (
        f"exp052_gap_generated_rationale_{cfg.exp_tag}/fold_{cfg.fold:02d}"
    )
    output_dir = env.artifacts_dir / exp_name

    LOGGER = get_logger(__name__, output_dir)
    seed_everything(cfg.seed)
    LOGGER.info(f"Experiment: {exp_name}")
    LOGGER.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # Resolve data path
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
        f"Fold {cfg.fold}: train={len(split_dict['train'])}, "
        f"val={len(split_dict['val'])}"
    )

    # Augmentation pipeline (mirror exp034 a00)
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

    from diema.data.collate import MotionDataModule
    body_scale_range = None
    if cfg.body_scale_jitter_lo > 0 and cfg.body_scale_jitter_hi > 0:
        body_scale_range = (
            float(cfg.body_scale_jitter_lo), float(cfg.body_scale_jitter_hi)
        )
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

    # Build backbone via registry, then wrap with PartAlign head
    from types import SimpleNamespace
    from diema.models import build_model

    backbone_cfg = SimpleNamespace(
        model=SimpleNamespace(
            name=cfg.model_name, num_class=cfg.num_class,
            in_channels=cfg.in_channels, dim=cfg.dim,
            num_conv_blocks=cfg.num_conv_blocks,
            kernel_size=cfg.kernel_size,
            num_cross_blocks=cfg.num_cross_blocks,
            num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
            drop_rate=cfg.drop_rate,
            late_dropout=cfg.late_dropout,
            late_dropout_start_step=cfg.late_dropout_start_step,
            use_per_part_gate=cfg.use_per_part_gate,
        ),
        skeleton=SimpleNamespace(
            num_nodes=cfg.num_nodes, inward_edges=list(cfg.inward_edges),
        ),
    )
    backbone = build_model(backbone_cfg)

    from diema.models.conv1d_transformer.partalign_conv1d_tr_model import (
        PartAlignConv1DTr_Model,
    )
    model = PartAlignConv1DTr_Model(
        backbone=backbone,
        text_dim=cfg.text_dim,
        num_parts=cfg.num_parts,
        proj_hidden=cfg.proj_hidden,
    )
    n_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"Model parameters: {n_params:,} (PartAlign mode)")

    # Rationale cache
    rationale_cache_path = Path(cfg.rationale_cache_path)
    if not rationale_cache_path.is_absolute():
        rationale_cache_path = env.project_root / rationale_cache_path
    if not rationale_cache_path.exists():
        LOGGER.error(
            f"Rationale cache not found: {rationale_cache_path}. "
            "Build it first with tools/build_rationale_text_cache.py "
            "(requires sweep output)."
        )
        return

    from diema.features.rationale_text_cache import RationaleTextCache
    rationale_cache = RationaleTextCache(str(rationale_cache_path))
    cov = rationale_cache.coverage_stats()
    LOGGER.info(
        f"Rationale cache: N={rationale_cache.num_rows} rows, "
        f"P={rationale_cache.num_parts} parts, D={rationale_cache.embedding_dim}, "
        f"per-part mask frac={['%.3f' % x for x in cov['per_part']]}, "
        f"global mask frac={cov['true_fraction']:.3f}"
    )

    loss_kwargs = {}
    if cfg.label_smoothing > 0:
        loss_kwargs["label_smoothing"] = cfg.label_smoothing

    from diema.training.part_align_trainer import PartAlignLightningModel
    filename_to_part_idx = {
        stem: idx for stem, idx in rationale_cache._stem_to_idx.items()
    }
    lit_model = PartAlignLightningModel(
        model=model,
        base_lr=cfg.lr,
        num_class=cfg.num_class,
        filename_to_part_idx=filename_to_part_idx,
        part_embedding_tensor=rationale_cache._embeddings,
        part_mask_tensor=rationale_cache._part_mask,
        part_loss_weight=cfg.part_loss_weight,
        part_loss_mode=cfg.part_loss_mode,
        part_loss_temperature=cfg.part_loss_temperature,
        part_loss_warmup_epochs=cfg.part_loss_warmup_epochs,
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
    from diema.training.callbacks import (
        MetricsSaveCallback, ConfigSaveCallback, OOFPredictionCallback,
    )
    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir), filename="best",
            monitor="val_f1", mode="max", save_top_k=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=cfg.early_stopping_patience,
            mode="min",
        ),
        MetricsSaveCallback(output_dir=output_dir),
        ConfigSaveCallback(
            config=OmegaConf.to_container(cfg, resolve=True),
            output_dir=output_dir,
        ),
        OOFPredictionCallback(
            output_dir=output_dir, monitor="val_f1", mode="max",
        ),
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

    # Same PyTorch 2.6 weights_only workaround as exp051 (see that file for
    # the long-form explanation). Load the best checkpoint manually then test.
    best_path = None
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            best_path = cb.best_model_path
            break
    if best_path and Path(best_path).exists():
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        lit_model.load_state_dict(ckpt["state_dict"])
        LOGGER.info(f"Restored best checkpoint from {best_path}")
    else:
        LOGGER.warning(
            "best checkpoint path not found; testing last-epoch weights"
        )
    trainer.test(lit_model, datamodule=datamodule)

    best_f1 = trainer.callback_metrics.get("val_f1", 0)
    best_acc = trainer.callback_metrics.get("val_acc", 0)
    LOGGER.info(f"Best val_f1: {best_f1:.4f}, val_acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
