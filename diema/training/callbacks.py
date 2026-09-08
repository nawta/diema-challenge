"""Training callbacks for OOF saving, metrics logging, etc.

See: diema_challenge_implementation_spec.md §6.5 — 保存物
Related: diema/training/train.py, diema/training/evaluate.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class MetricsSaveCallback(Callback):
    """Save validation and test metrics to JSON.

    Saves metrics.json with best_val_acc, test results, and per-epoch history.

    Args:
        output_dir: directory to save metrics.json
    """

    def __init__(self, output_dir: str | Path):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_acc = 0.0
        self.history = []
        self.test_results = {}

    def _save(self):
        output_path = self.output_dir / "metrics.json"
        data = {
            "best_val_acc": self.best_val_acc,
            "test_results": self.test_results,
            "history": self.history,
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = {k: v.item() if hasattr(v, "item") else v for k, v in trainer.callback_metrics.items()}
        self.history.append({"epoch": trainer.current_epoch, **metrics})

        val_acc = metrics.get("val_acc", 0.0)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

        self._save()

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = {k: v.item() if hasattr(v, "item") else v for k, v in trainer.callback_metrics.items()}
        self.test_results = {
            "test_acc": metrics.get("test_acc", 0.0),
            "test_f1": metrics.get("test_f1", 0.0),
        }
        self._save()
        logger.info(f"Test results: acc={self.test_results['test_acc']:.4f}, f1={self.test_results['test_f1']:.4f}")


class OOFPredictionCallback(Callback):
    """Save validation set logits and labels for the best epoch.

    Used by ensemble pipelines to avoid retraining models.
    Saves on each new-best validation metric (val_f1 by default).

    Output files (under output_dir):
        oof_logits.npy   : (N_val, num_class) float32
        oof_labels.npy   : (N_val,) int64
        oof_filenames.txt: list of filenames, one per line

    Args:
        output_dir: directory to save OOF artifacts
        monitor: metric name to track (default: "val_f1")
        mode: "max" or "min"
    """

    def __init__(self, output_dir: str | Path, monitor: str = "val_f1", mode: str = "max"):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_metric = -float("inf") if mode == "max" else float("inf")
        self._buffer_logits = []
        self._buffer_labels = []
        self._buffer_filenames = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Re-run forward to capture logits (validation_step doesn't return them)
        inputs, labels, names = batch
        with torch.no_grad():
            out = pl_module(inputs)
        self._buffer_logits.append(out["logits"].detach().cpu().float().numpy())
        self._buffer_labels.append(labels.detach().cpu().numpy())
        self._buffer_filenames.extend(list(names))

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = trainer.callback_metrics
        current = metrics.get(self.monitor)
        if current is None:
            self._reset_buffers()
            return
        current = current.item() if hasattr(current, "item") else float(current)

        improved = (current > self.best_metric) if self.mode == "max" else (current < self.best_metric)
        if improved:
            self.best_metric = current
            if self._buffer_logits:
                logits = np.concatenate(self._buffer_logits, axis=0)
                labels = np.concatenate(self._buffer_labels, axis=0)
                np.save(self.output_dir / "oof_logits.npy", logits)
                np.save(self.output_dir / "oof_labels.npy", labels)
                with open(self.output_dir / "oof_filenames.txt", "w") as f:
                    f.write("\n".join(self._buffer_filenames))
                logger.info(
                    f"OOF saved at {self.monitor}={current:.4f}: "
                    f"{logits.shape[0]} samples, {logits.shape[1]} classes"
                )

        self._reset_buffers()

    def _reset_buffers(self):
        self._buffer_logits = []
        self._buffer_labels = []
        self._buffer_filenames = []


class ConfigSaveCallback(Callback):
    """Save experiment config to YAML at the start of training.

    Args:
        config: config dict or namespace to save
        output_dir: directory to save config.yaml
    """

    def __init__(self, config: dict, output_dir: str | Path):
        super().__init__()
        self.config = config
        self.output_dir = Path(output_dir)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "config.yaml"

        import yaml
        config_dict = self.config if isinstance(self.config, dict) else vars(self.config)
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Config saved to {output_path}")
