"""exp100: GroupDRO + Region-Aware Conv1D + Transformer (Phase ACC Track B).

GroupDRO (Sagawa et al. 2020, https://arxiv.org/abs/1911.08731) on top of
the exp034 region-aware backbone. Reweights per-sample CE toward the worst-
performing performer group (JP / TW) to fight the diagnosed performer confound
(performer effect = 10.8× emotion effect, η² ratio).

GroupDRO adds NO new parameters — only reweights CE — which avoids the
backbone-collapse failure modes of prior auxiliary-loss approaches:
  - DANN/GRL (exp036): −2.65pt
  - SupCon (exp040): −11.53pt
  - MoE (exp042): −18pt

Default settings (Track B a00):
  - dim=160, AdamW lr=4e-3, cosine_warmup (warmup 5 / 80 ep)
  - label_smoothing=0.3, drop_rate=0.3 (matches exp034-a00)
  - GroupDRO: group_mode="country", eta_dro=0.01, warmup_epochs_dro=10

See: Phase ACC Track B — B0
Plan: ~/.claude/plans/replicated-gathering-raven.md Track B (GroupDRO)
Related: diema/training/group_dro_trainer.py::GroupDROLightningModel
         experiments/exp034_regionaware_convtr/run.py  (base pattern)

Usage:
    python -m experiments.exp100_groupdro_regionaware.run fold=0
    python -m experiments.exp100_groupdro_regionaware.run fold=0 max_epochs=5
    python -m experiments.exp100_groupdro_regionaware.run fold=0 eta_dro=0.05
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
    stream_type: str = "rotation_6d"
    clip_length: int = 64
    num_workers: int = 8

    # model (mirrors exp034-a00)
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

    # skeleton
    num_nodes: int = 25
    inward_edges: list = field(default_factory=lambda: list(DIEMA_INWARD_EDGES))
    lr_joint_pairs: list = field(default_factory=lambda: list(DIEMA_LR_PAIRS_BVH))

    # training (mirrors exp034-a00)
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

    # augmentation (same as exp034-a00)
    augment_rotate: bool = True
    augment_mirror: bool = True
    augment_speed: bool = True
    augment_noise_sigma: float = 2.5

    # domain generalization (opt-in; off by default)
    seq_cutout_p: float = 0.0
    seq_cutout_segments: int = 5
    seq_cutout_ratio: float = 0.15
    body_scale_jitter_lo: float = 0.0
    body_scale_jitter_hi: float = 0.0
    cutmix_alpha: float = 0.0
    # mixup_alpha is always 0.0 for GroupDRO — GroupDROLightningModel raises
    # if you try to pass mixup_alpha > 0. Not exposed here to prevent accidents.

    # GroupDRO-specific settings
    use_group_dro: bool = True
    group_mode: str = "country"       # "country" → JP=0, TW=1
    num_groups: int = 2
    eta_dro: float = 0.01             # DRO dual LR (Sagawa 2020 default)
    warmup_epochs_dro: int = 10       # plain CE warmup before DRO activates


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    exp_cfg = OmegaConf.structured(ExpConfig)
    known_keys = set(OmegaConf.to_container(exp_cfg).keys())
    filtered = {k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items()
                if k in known_keys}
    cfg = OmegaConf.merge(exp_cfg, OmegaConf.create(filtered))

    env = EnvConfig()
    exp_name = f"exp100_groupdro_regionaware_{cfg.exp_tag}/fold_{cfg.fold:02d}"
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

    # Augmentation pipeline (mirrors exp034)
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

    # DataModule (stream_type=rotation_6d, in_channels=6)
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

    # Fail-fast in_channels check (mirrors exp034 guard)
    expected_ch = MotionDataModule.expected_in_channels(cfg.stream_type, None)
    if cfg.in_channels != expected_ch:
        raise ValueError(
            f"in_channels mismatch: cfg.in_channels={cfg.in_channels} but "
            f"stream_type='{cfg.stream_type}' expects {expected_ch} channels."
        )
    LOGGER.info(f"stream_type='{cfg.stream_type}', in_channels={cfg.in_channels} (OK)")

    # Model (same build as exp034)
    from types import SimpleNamespace
    from diema.models import build_model
    model = build_model(SimpleNamespace(
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

    n_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    loss_kwargs = {}
    if cfg.label_smoothing > 0:
        loss_kwargs["label_smoothing"] = cfg.label_smoothing

    # Lightning model: GroupDRO or plain CE (use_group_dro flag)
    if cfg.use_group_dro:
        from diema.training.group_dro_trainer import GroupDROLightningModel
        LOGGER.info(
            f"GroupDRO mode: group_mode={cfg.group_mode}, num_groups={cfg.num_groups}, "
            f"eta_dro={cfg.eta_dro}, warmup_epochs_dro={cfg.warmup_epochs_dro}"
        )
        lit_model = GroupDROLightningModel(
            model=model,
            base_lr=cfg.lr,
            num_class=cfg.num_class,
            group_mode=cfg.group_mode,
            num_groups=cfg.num_groups,
            eta_dro=cfg.eta_dro,
            warmup_epochs_dro=cfg.warmup_epochs_dro,
            loss_type=cfg.loss_type,
            loss_kwargs=loss_kwargs,
            optimizer=cfg.optimizer,
            scheduler_type=cfg.scheduler_type,
            weight_decay=cfg.weight_decay,
            warmup_epochs=cfg.warmup_epochs,
            cutmix_alpha=cfg.cutmix_alpha,
        )
    else:
        # Fallback: plain CE (useful for ablation comparison without reweighting)
        from diema.training.train import LightningModel
        LOGGER.info("Plain CE mode (use_group_dro=false, ablation/baseline)")
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
            cutmix_alpha=cfg.cutmix_alpha,
        )

    # Callbacks: monitor val_f1 (primary metric — same as exp034)
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
    trainer.test(lit_model, datamodule=datamodule, ckpt_path="best")

    LOGGER.info(f"Done. Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
