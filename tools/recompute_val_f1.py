"""Recompute val_f1 + per-fold confusion matrix from existing best.ckpt files.

For exp001/exp002 experiments that were trained before val_f1 was added.
Uses the same fold split as the original training.

Output: <exp_dir>/<fold>/val_metrics.json with:
  - val_acc, val_f1
  - confusion_matrix (12x12)
  - per_class_f1 (length 12)

Usage:
    python tools/recompute_val_f1.py --experiment exp001_benchmark_repro
    python tools/recompute_val_f1.py --experiment exp002_a04_smooth
"""

import json
from pathlib import Path

import click
import numpy as np
import torch

from utils.env import EnvConfig


@click.command()
@click.option("--experiment", required=True, help="Experiment folder name under output/artifacts/")
@click.option("--num-folds", type=int, default=10, help="Number of folds")
@click.option("--num-class", type=int, default=12, help="Number of classes")
def main(experiment: str, num_folds: int, num_class: int):
    import pybvh_ml
    from torchmetrics.functional import accuracy, confusion_matrix, f1_score

    from diema.data.collate import motion_collate_fn
    from diema.data.dataset import MotionDataset
    from diema.data.splits import generate_lpo_splits
    from diema.models import build_model
    from diema.training.train import LightningModel

    env = EnvConfig()
    exp_dir = env.artifacts_dir / experiment

    if not exp_dir.exists():
        click.echo(f"ERROR: experiment dir not found: {exp_dir}", err=True)
        raise SystemExit(1)

    # Load preprocessed data once and generate splits (same as run.py)
    data_path = env.processed_dir / "motion_train_quat.npz"
    preprocessed = pybvh_ml.load_preprocessed(str(data_path))
    filenames = preprocessed["filenames"]
    splits = generate_lpo_splits(filenames, num_folds=num_folds)

    summary = {"folds": []}

    for fold in range(num_folds):
        fold_dir = exp_dir / f"fold_{fold:02d}"
        ckpt_path = fold_dir / "best.ckpt"
        cfg_path = fold_dir / "config.yaml"
        if not ckpt_path.exists() or not cfg_path.exists():
            click.echo(f"  fold_{fold:02d}: SKIP (missing ckpt or config)")
            continue

        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        # Build model
        from types import SimpleNamespace
        model = build_model(SimpleNamespace(
            model=SimpleNamespace(
                name=cfg.get("model_name", "stgcn"),
                num_class=cfg.get("num_class", 12),
                in_channels=cfg.get("in_channels", 6),
                dropout=cfg.get("dropout", 0.5),
                edge_weighting=cfg.get("edge_weighting", True),
                plusplus=cfg.get("plusplus", True),
                unit_dropout=cfg.get("unit_dropout", 0.15),
            ),
            skeleton=SimpleNamespace(
                num_nodes=cfg.get("num_nodes", 25),
                inward_edges=cfg.get("inward_edges"),
            ),
        ))

        lit_model = LightningModel(
            model=model,
            base_lr=cfg.get("lr", 0.2),
            num_class=cfg.get("num_class", 12),
            loss_type=cfg.get("loss_type", "ce"),
            optimizer=cfg.get("optimizer", "SGD"),
            scheduler_type=cfg.get("scheduler_type", "cosine"),
            weight_decay=cfg.get("weight_decay", 5e-4),
        )

        # Load checkpoint
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        lit_model.load_state_dict(state["state_dict"])
        lit_model.eval()
        lit_model.cuda()

        # Build val dataset (same split as training)
        split_dict = splits[fold]
        val_indices = [idx for _, idx in split_dict["val"]]
        ds = MotionDataset(
            data_path=str(data_path),
            indices=val_indices,
            clip_length=cfg.get("clip_length", 64),
            target_repr=cfg.get("target_repr", "6d"),
            is_test=True,
            seed=cfg.get("seed", 42),
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=cfg.get("batch_size", 128), shuffle=False,
            num_workers=4, collate_fn=motion_collate_fn,
        )

        all_logits = []
        all_labels = []
        all_filenames = []
        with torch.no_grad():
            for data, labels, names in loader:
                out = lit_model(data.cuda())
                all_logits.append(out["logits"].cpu().float())
                all_labels.append(labels)
                all_filenames.extend(list(names))
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        preds = logits.argmax(dim=1)

        acc = accuracy(preds, labels, task="multiclass", num_classes=num_class).item()
        f1 = f1_score(preds, labels, task="multiclass", num_classes=num_class, average="macro").item()
        per_class_f1 = f1_score(
            preds, labels, task="multiclass", num_classes=num_class, average="none"
        ).cpu().numpy().tolist()
        cm = confusion_matrix(preds, labels, task="multiclass", num_classes=num_class).cpu().numpy().tolist()

        # Save val_metrics.json + OOF
        out = {
            "val_acc": acc,
            "val_f1": f1,
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm,
        }
        with open(fold_dir / "val_metrics.json", "w") as f:
            json.dump(out, f, indent=2)
        np.save(fold_dir / "oof_logits.npy", logits.numpy().astype(np.float32))
        np.save(fold_dir / "oof_labels.npy", labels.numpy().astype(np.int64))
        with open(fold_dir / "oof_filenames.txt", "w") as f:
            f.write("\n".join(all_filenames))

        click.echo(f"  fold_{fold:02d}: val_acc={acc:.4f}, val_f1={f1:.4f}, n={len(labels)}")

        summary["folds"].append({
            "fold": fold,
            "val_acc": acc,
            "val_f1": f1,
            "n_val": len(labels),
        })

    # Aggregate
    if summary["folds"]:
        accs = [f["val_acc"] for f in summary["folds"]]
        f1s = [f["val_f1"] for f in summary["folds"]]
        summary["mean_val_acc"] = float(np.mean(accs))
        summary["std_val_acc"] = float(np.std(accs))
        summary["mean_val_f1"] = float(np.mean(f1s))
        summary["std_val_f1"] = float(np.std(f1s))

        with open(exp_dir / "val_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        click.echo(
            f"\n  Mean val_acc = {summary['mean_val_acc']*100:.2f}% ± {summary['std_val_acc']*100:.2f}%"
        )
        click.echo(
            f"  Mean val_f1  = {summary['mean_val_f1']*100:.2f}% ± {summary['std_val_f1']*100:.2f}%"
        )


if __name__ == "__main__":
    main()
