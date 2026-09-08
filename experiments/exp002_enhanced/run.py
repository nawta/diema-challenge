"""exp002: Enhanced STGCN++ with longer clips, label smoothing, multi-clip test.

Improvements over exp001:
  - clip_length: 64 → 128/256 (capture more temporal context)
  - label_smoothing: 0.1 (soften ambiguous emotion boundaries)
  - test_clips: multi-clip test-time averaging for stable predictions
  - max_epochs: 100 (longer training with patience)

See: docs/experiments.md, experiments/exp001_benchmark_repro
Related: diema/models/stgcn/, diema/training/train.py

Usage:
    python -m experiments.exp002_enhanced.run exp=a00
    python -m experiments.exp002_enhanced.run clip_length=256 fold=0
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
import numpy as np
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
    data_path: str = ""
    target_repr: str = "6d"
    clip_length: int = 128       # longer than exp001 (64)
    num_workers: int = 8

    # model
    model_name: str = "stgcn"
    num_class: int = 12
    in_channels: int = 6
    dropout: float = 0.5
    edge_weighting: bool = True
    plusplus: bool = True
    unit_dropout: float = 0.15

    # skeleton (DIEM-A 25 nodes, CTV node indices)
    num_nodes: int = 25
    inward_edges: list = field(default_factory=lambda: [
        [0, 1], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7],
        [9, 5], [10, 9], [11, 10], [12, 11],
        [13, 5], [14, 13], [15, 14], [16, 15],
        [17, 1], [18, 17], [19, 18], [20, 19],
        [21, 1], [22, 21], [23, 22], [24, 23],
    ])

    # training
    batch_size: int = 256        # larger batch (GPU has headroom)
    max_epochs: int = 100        # longer training
    optimizer: str = "SGD"
    lr: float = 0.2
    weight_decay: float = 5.0e-4
    scheduler_type: str = "cosine"
    early_stopping_patience: int = 15   # more patience for longer clips
    loss_type: str = "ce"
    label_smoothing: float = 0.1        # new: soften targets

    # test-time
    test_clips: int = 10         # new: multi-clip test inference

    # experiment tag (included in output path for parallel runs)
    exp_tag: str = "a00"

    # augmentation
    augment_rotate: bool = True
    augment_mirror: bool = True
    augment_speed: bool = True
    augment_noise_sigma: float = 2.5

    # lr_joint_pairs (CTV node indices)
    lr_joint_pairs: list = field(default_factory=lambda: [
        [12, 8], [13, 9], [14, 10], [15, 11],
        [20, 16], [21, 17], [22, 18], [23, 19],
    ])


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """Run single-fold training with enhancements."""
    exp_cfg = OmegaConf.structured(ExpConfig)
    known_keys = set(OmegaConf.to_container(exp_cfg).keys())
    filtered = {k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items()
                if k in known_keys}
    cfg = OmegaConf.merge(exp_cfg, OmegaConf.create(filtered))

    env = EnvConfig()
    exp_name = f"exp002_{cfg.exp_tag}/fold_{cfg.fold:02d}"
    output_dir = env.artifacts_dir / exp_name

    LOGGER = get_logger(__name__, output_dir)
    seed_everything(cfg.seed)

    LOGGER.info(f"Experiment: {exp_name}")
    LOGGER.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # Data path
    if cfg.data_path:
        data_path = Path(cfg.data_path)
        if not data_path.is_absolute():
            data_path = env.project_root / data_path
    else:
        data_path = env.processed_dir / "motion_train_quat.npz"
    if not data_path.exists():
        LOGGER.error(f"Data not found: {data_path}. Run 'make prepare-data' first.")
        return

    # Load data + split
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

    # Model
    from types import SimpleNamespace
    from diema.models import build_model
    model = build_model(SimpleNamespace(
        model=SimpleNamespace(
            name=cfg.model_name, num_class=cfg.num_class, in_channels=cfg.in_channels,
            dropout=cfg.dropout, edge_weighting=cfg.edge_weighting,
            plusplus=cfg.plusplus, unit_dropout=cfg.unit_dropout,
        ),
        skeleton=SimpleNamespace(num_nodes=cfg.num_nodes, inward_edges=list(cfg.inward_edges)),
    ))

    # Loss kwargs
    loss_kwargs = {}
    if cfg.label_smoothing > 0:
        loss_kwargs["label_smoothing"] = cfg.label_smoothing

    # Lightning model
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
    )

    # Callbacks
    from diema.training.callbacks import MetricsSaveCallback, ConfigSaveCallback
    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir), filename="best",
            monitor="val_acc", mode="max", save_top_k=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=cfg.early_stopping_patience, mode="min",
        ),
        MetricsSaveCallback(output_dir=output_dir),
        ConfigSaveCallback(config=OmegaConf.to_container(cfg, resolve=True), output_dir=output_dir),
    ]

    # Loggers
    loggers = [CSVLogger(save_dir=str(output_dir), name="csv_logs")]
    try:
        import wandb
        loggers.append(WandbLogger(
            project=WANDB_PROJECT_NAME, name=exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        ))
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

    # Test: multi-clip averaging.
    # Read the actual best checkpoint path from ModelCheckpoint (not a
    # stale best.ckpt left behind by a previous smoke run in the same
    # output dir).
    LOGGER.info(f"Running test evaluation (test_clips={cfg.test_clips})...")
    best_model_path = getattr(trainer.checkpoint_callback, "best_model_path", "")
    ckpt_path = best_model_path if best_model_path else None

    if cfg.test_clips > 1:
        # Multi-clip test: run test N times and average logits
        _run_multi_clip_test(
            trainer, lit_model, datamodule, ckpt_path,
            cfg.test_clips, output_dir, LOGGER,
        )
    else:
        trainer.test(
            lit_model, datamodule=datamodule,
            ckpt_path=(ckpt_path if ckpt_path is not None else "best"),
        )

    LOGGER.info(f"Done. Artifacts saved to {output_dir}")


def _run_multi_clip_test(trainer, lit_model, datamodule, ckpt_path, num_clips, output_dir, logger):
    """Run test N times with different random temporal crops, average logits."""
    import json
    from diema.data.dataset import MotionDataset
    from diema.data.collate import motion_collate_fn

    # Load best weights
    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu")
        lit_model.load_state_dict(state["state_dict"])

    lit_model.eval()
    lit_model.cuda()

    # Build test dataset with different seeds
    test_indices = datamodule.test_indices
    all_logits = []

    for clip_idx in range(num_clips):
        ds = MotionDataset(
            data_path=str(datamodule.data_path),
            indices=test_indices,
            clip_length=datamodule.clip_length,
            target_repr=datamodule.target_repr,
            is_test=False,   # use random crop (not center) for diversity
            seed=42 + clip_idx,
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=datamodule.batch_size, shuffle=False,
            num_workers=0, collate_fn=motion_collate_fn,
        )

        clip_logits = []
        clip_labels = []
        with torch.no_grad():
            for data, labels, _ in loader:
                out = lit_model(data.cuda())
                clip_logits.append(out["logits"].cpu())
                clip_labels.append(labels)

        all_logits.append(torch.cat(clip_logits, dim=0))
        if clip_idx == 0:
            all_labels = torch.cat(clip_labels, dim=0)

    # Average logits
    avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)
    preds = avg_logits.argmax(dim=1)

    from torchmetrics.functional import accuracy, f1_score
    acc = accuracy(preds, all_labels, task="multiclass", num_classes=12).item()
    f1 = f1_score(preds, all_labels, task="multiclass", num_classes=12, average="macro").item()

    logger.info(f"Multi-clip test ({num_clips} clips): acc={acc:.4f}, f1={f1:.4f}")

    # Save to metrics.json
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}
    metrics["test_results"] = {"test_acc": acc, "test_f1": f1, "test_clips": num_clips}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


if __name__ == "__main__":
    main()
