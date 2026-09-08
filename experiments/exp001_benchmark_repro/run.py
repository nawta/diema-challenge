"""exp001: STGCN++ benchmark reproduction.

Reproduces the baseline STGCN++ 10-fold LPO results from the
diema-challenge-benchmark. Uses the same architecture and hyperparameters.

See: diema_challenge_implementation_spec.md §8
Related: diema/models/stgcn/, diema/training/train.py, tools/run_cv.py

Usage:
    python -m experiments.exp001_benchmark_repro.run exp=000
    python -m experiments.exp001_benchmark_repro.run exp=000 fold=0 num_folds=10
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

WANDB_PROJECT_NAME = "diema-challenge"


@dataclass
class ExpConfig:
    """Experiment configuration."""
    debug: bool = False
    seed: int = 42
    fold: int = 0
    num_folds: int = 10

    # data
    data_path: str = ""  # resolved from EnvConfig.processed_dir if empty
    target_repr: str = "6d"
    clip_length: int = 64
    num_workers: int = 8

    # model
    model_name: str = "stgcn"
    num_class: int = 12
    in_channels: int = 6
    dropout: float = 0.5
    edge_weighting: bool = True
    plusplus: bool = True
    unit_dropout: float = 0.15

    # skeleton (DIEM-A 25 nodes: node 0=root_pos, nodes 1-24=BVH joints)
    # Matches diema-challenge-benchmark/configs/diema12_stgcn.yaml
    num_nodes: int = 25
    inward_edges: list = field(default_factory=lambda: [
        [0, 1], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7],
        [9, 5], [10, 9], [11, 10], [12, 11],
        [13, 5], [14, 13], [15, 14], [16, 15],
        [17, 1], [18, 17], [19, 18], [20, 19],
        [21, 1], [22, 21], [23, 22], [24, 23],
    ])

    # training
    batch_size: int = 128
    max_epochs: int = 65
    optimizer: str = "SGD"
    lr: float = 0.2
    weight_decay: float = 5.0e-4
    scheduler_type: str = "cosine"
    early_stopping_patience: int = 10
    loss_type: str = "ce"

    # augmentation
    augment_rotate: bool = True
    augment_mirror: bool = True
    augment_speed: bool = True
    augment_noise_sigma: float = 2.5

    # lr_joint_pairs for mirror augmentation (CTV node indices, from benchmark config)
    lr_joint_pairs: list = field(default_factory=lambda: [
        [12, 8], [13, 9], [14, 10], [15, 11],
        [20, 16], [21, 17], [22, 18], [23, 19],
    ])


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """Run single-fold training for benchmark reproduction."""
    # Merge Hydra config with ExpConfig defaults (drop unknown keys)
    exp_cfg = OmegaConf.structured(ExpConfig)
    known_keys = set(OmegaConf.to_container(exp_cfg).keys())
    filtered = {k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items()
                if k in known_keys}
    cfg = OmegaConf.merge(exp_cfg, OmegaConf.create(filtered))

    env = EnvConfig()
    exp_name = f"exp001_benchmark_repro/fold_{cfg.fold:02d}"
    output_dir = env.artifacts_dir / exp_name

    LOGGER = get_logger(__name__, output_dir)
    seed_everything(cfg.seed)

    LOGGER.info(f"Experiment: {exp_name}")
    LOGGER.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # Data path resolution
    if cfg.data_path:
        data_path = Path(cfg.data_path)
        if not data_path.is_absolute():
            data_path = env.project_root / data_path
    else:
        data_path = env.processed_dir / "motion_train_quat.npz"
    if not data_path.exists():
        LOGGER.error(f"Data not found: {data_path}. Run 'make prepare-data' first.")
        return

    # Load preprocessed data and generate split
    preprocessed = pybvh_ml.load_preprocessed(str(data_path))
    filenames = preprocessed["filenames"]

    from diema.data.splits import generate_lpo_splits
    splits = generate_lpo_splits(filenames, num_folds=cfg.num_folds)
    split_dict = splits[cfg.fold]

    LOGGER.info(f"Fold {cfg.fold}: train={len(split_dict['train'])}, val={len(split_dict['val'])}")

    # Build augmentation pipeline
    # Each entry: (fn, probability, kwargs)
    # kwargs values can be callables: lambda rng -> value (sampled per sample)
    aug_pipeline = None
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
    if aug_steps:
        aug_pipeline = pybvh_ml.AugmentationPipeline(aug_steps)

    # Build DataModule
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

    # Build model
    from types import SimpleNamespace
    model_config = SimpleNamespace(
        model=SimpleNamespace(
            num_class=cfg.num_class,
            in_channels=cfg.in_channels,
            dropout=cfg.dropout,
            edge_weighting=cfg.edge_weighting,
            plusplus=cfg.plusplus,
            unit_dropout=cfg.unit_dropout,
        ),
        skeleton=SimpleNamespace(
            num_nodes=cfg.num_nodes,
            inward_edges=list(cfg.inward_edges),
        ),
    )

    from diema.models import build_model
    model = build_model(SimpleNamespace(model=SimpleNamespace(name=cfg.model_name, **vars(model_config.model)),
                                        skeleton=model_config.skeleton))

    # Build Lightning model
    from diema.training.train import LightningModel
    lit_model = LightningModel(
        model=model,
        base_lr=cfg.lr,
        num_class=cfg.num_class,
        loss_type=cfg.loss_type,
        optimizer=cfg.optimizer,
        scheduler_type=cfg.scheduler_type,
        weight_decay=cfg.weight_decay,
    )

    # Callbacks
    from diema.training.callbacks import MetricsSaveCallback, ConfigSaveCallback
    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir),
            filename="best",
            monitor="val_acc",
            mode="max",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping_patience,
            mode="min",
        ),
        MetricsSaveCallback(output_dir=output_dir),
        ConfigSaveCallback(config=OmegaConf.to_container(cfg, resolve=True), output_dir=output_dir),
    ]

    # Loggers
    loggers = [CSVLogger(save_dir=str(output_dir), name="csv_logs")]
    try:
        import wandb
        wandb_logger = WandbLogger(
            project=WANDB_PROJECT_NAME,
            name=exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        loggers.append(wandb_logger)
    except Exception:
        LOGGER.warning("wandb not available, using CSV logger only")

    # Trainer
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

    # Train
    LOGGER.info("Starting training...")
    trainer.fit(lit_model, datamodule=datamodule)

    # Test with best checkpoint — use Lightning's "best" sentinel so we
    # reload *this* run's best checkpoint (tracked by ModelCheckpoint),
    # not a stale best.ckpt left behind by a previous smoke run in the
    # same output dir.
    LOGGER.info("Running test evaluation...")
    trainer.test(lit_model, datamodule=datamodule, ckpt_path="best")

    LOGGER.info(f"Done. Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
