"""exp078 — per-class temporal+spatial saliency using exp034 a00 checkpoints.

For each LPO fold:
  1. Load best.ckpt
  2. Iterate val samples (one per call to avoid batching issues with grads)
  3. Compute |grad(target_logit) × input| as saliency (C, T, V)
  4. Accumulate per-class saliency sums + sample counts

After all folds, average per class and save:
  - per-class (T, V) joint × time heatmaps
  - per-class (T,) temporal salience
  - per-class (V,) spatial salience

See: experiments/exp078_temporal_spatial_saliency
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from diema.data.collate import MotionDataModule
from diema.data.parser import IDX_TO_EMOTION
from diema.models import build_model
from diema.models.skeleton_graph import DIEMA_INWARD_EDGES, DIEMA_JOINT_NAMES


def build_exp034_a00_model() -> torch.nn.Module:
    """Reproduce the model object built by exp034 run.py for a00."""
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            name="region_aware_conv1d_transformer",
            num_class=12,
            in_channels=6,
            clip_length=64,
            dim=160,
            num_conv_blocks=3,
            kernel_size=7,
            num_cross_blocks=2,
            num_heads=4,
            mlp_ratio=2.0,
            drop_rate=0.3,
            late_dropout=0.8,
            late_dropout_start_step=1000,
            use_per_part_gate=True,
        ),
        skeleton=SimpleNamespace(num_nodes=25, inward_edges=list(DIEMA_INWARD_EDGES)),
    )
    return build_model(cfg)


def strip_model_prefix(state_dict: dict) -> dict:
    """`LightningModel.state_dict` saves under `model.*` prefix; strip it."""
    return {k[len("model."):] if k.startswith("model.") else k: v for k, v in state_dict.items()}


def load_fold_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Strict load — assert checkpoint matches model architecture exactly.

    Strict loading (raise on any missing/unexpected key) verifies that the
    classifier head, BatchNorm running stats, and all learned parameters
    are loaded faithfully. A follow-up review flagged the previous
    `strict=False` with a `running_mean`-only filter as a silent
    invalidation risk for the paper figures — empirically benign for
    exp034 a00 (316/316 keys match) but switched to strict for safety.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = strip_model_prefix(ckpt["state_dict"])
    model = build_exp034_a00_model()
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-npz", default="data/diema_challenge/processed/motion_train_quat.npz")
    parser.add_argument("--splits-pkl", default="data/diema_challenge/processed/split_lpo_10fold.pkl")
    parser.add_argument("--ckpt-root", default="output/artifacts/exp034_regionaware_convtr_a00")
    parser.add_argument("--folds", default="all", help="'all' or comma-sep ids")
    parser.add_argument("--samples-per-fold", type=int, default=-1, help="-1 for all val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="experiments/exp078_temporal_spatial_saliency/results.npz")
    parser.add_argument("--target-repr", default="6d", choices=["6d"])
    parser.add_argument("--clip-length", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.folds == "all":
        fold_ids = list(range(10))
    else:
        fold_ids = [int(x) for x in args.folds.split(",")]

    with open(args.splits_pkl, "rb") as f:
        all_splits = pickle.load(f)

    NUM_CLASS = 12
    C, T, V = 6, args.clip_length, 25
    saliency_sum = np.zeros((NUM_CLASS, C, T, V), dtype=np.float64)
    count_per_class = np.zeros(NUM_CLASS, dtype=np.int64)

    t0 = time.time()
    for fid in fold_ids:
        ckpt_path = Path(args.ckpt_root) / f"fold_{fid:02d}" / "best.ckpt"
        if not ckpt_path.exists():
            print(f"[skip] fold {fid}: no checkpoint at {ckpt_path}")
            continue
        print(f"[fold {fid:02d}] loading {ckpt_path}")
        model = load_fold_model(ckpt_path, device)

        dm = MotionDataModule(
            data_path=args.motion_npz,
            split_dict=all_splits[fid],
            clip_length=args.clip_length,
            batch_size=args.batch_size,
            num_workers=0,
            target_repr=args.target_repr,
            augmentation_pipeline=None,
            debug=False,
        )
        dm.setup("fit")
        val_loader = dm.val_dataloader()

        n_processed = 0
        for batch in val_loader:
            if isinstance(batch, dict):
                x = batch["motion"].to(device)
                y = batch["label"].to(device)
            elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x = batch[0].to(device)
                y = batch[1].to(device)
            else:
                raise TypeError(f"unexpected batch type: {type(batch)}")
            x = x.detach().clone().requires_grad_(True)

            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            out = model(x)
            logits = out["logits"] if isinstance(out, dict) else out
            target_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
            target_logits.sum().backward()

            sal = (x.grad * x).detach().abs().cpu().numpy()  # (N, C, T, V)
            y_np = y.cpu().numpy()
            for i, c in enumerate(y_np):
                saliency_sum[int(c)] += sal[i]
                count_per_class[int(c)] += 1
            n_processed += x.shape[0]
            if args.samples_per_fold > 0 and n_processed >= args.samples_per_fold:
                break

        del model
        torch.cuda.empty_cache()
        print(f"[fold {fid:02d}] processed {n_processed} val samples, total elapsed {time.time()-t0:.1f}s")

    saliency_mean = np.zeros_like(saliency_sum)
    for c in range(NUM_CLASS):
        if count_per_class[c] > 0:
            saliency_mean[c] = saliency_sum[c] / count_per_class[c]

    temporal_sal = saliency_mean.sum(axis=(1, 3))  # (12, T)
    spatial_sal = saliency_mean.sum(axis=(1, 2))   # (12, V)
    joint_time_sal = saliency_mean.sum(axis=1)     # (12, T, V)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        saliency_mean=saliency_mean.astype(np.float32),  # (12, C, T, V)
        temporal_sal=temporal_sal.astype(np.float32),    # (12, T)
        spatial_sal=spatial_sal.astype(np.float32),      # (12, V)
        joint_time_sal=joint_time_sal.astype(np.float32),# (12, T, V)
        count_per_class=count_per_class,
        emotion_labels=np.array([IDX_TO_EMOTION[c] for c in range(NUM_CLASS)]),
        joint_names=np.array(DIEMA_JOINT_NAMES),
    )
    print(f"[exp078] saved {args.output}")
    print(f"[exp078] per-class counts: {dict(zip([IDX_TO_EMOTION[c] for c in range(NUM_CLASS)], count_per_class.tolist()))}")
    print(f"[exp078] total elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
