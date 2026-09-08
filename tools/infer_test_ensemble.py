"""Run test inference with the 7-way equal-weight ensemble.

For each ensemble member (experiment), loads all 10 fold checkpoints,
runs inference on the test NPZ, and averages the fold predictions.
Then averages across experiments (equal weights, per decision D6/D7).
Produces a submission CSV.

Usage::

    python tools/infer_test_ensemble.py \
      --experiments exp002_a04_smooth \
      --experiments exp003_ctrgcn_a00 \
      ...

See: docs/decisions.md D6, D7.
Related: tools/make_submission.py, tools/check_submission.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import click
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pybvh_ml  # noqa: E402
from utils.env import EnvConfig  # noqa: E402
from diema.data.parser import IDX_TO_EMOTION  # noqa: E402
from diema.models import build_model  # noqa: E402
import diema.models  # noqa: F401, E402  (register models)


def _cfg_to_builder_ns(cfg_dict: dict) -> SimpleNamespace:
    """Convert a saved config.yaml dict to the SimpleNamespace tree that
    ``build_model`` expects.

    Each ``from_config`` classmethod uses ``getattr(config.model, key, default)``
    to read model-specific kwargs, so we simply forward all flat-level keys
    into ``config.model`` as-is. Unknown keys are ignored by from_config; any
    missing keys fall back to the model's own defaults.
    """
    name = cfg_dict["model_name"]
    skeleton = SimpleNamespace(
        num_nodes=cfg_dict.get("num_nodes", 25),
        inward_edges=list(cfg_dict.get("inward_edges", [])),
    )
    # Forward everything that looks like a model-relevant key into config.model.
    # We include the known common keys, and pass through any additional keys
    # that appear flat in the YAML (each model's from_config picks what it
    # needs via getattr with a default).
    skip_keys = {
        "debug", "seed", "fold", "num_folds", "exp_tag", "data_path",
        "target_repr", "clip_length", "num_workers",
        "num_nodes", "inward_edges", "lr_joint_pairs",
        "batch_size", "max_epochs", "optimizer", "lr", "weight_decay",
        "scheduler_type", "warmup_epochs", "early_stopping_patience",
        "loss_type", "label_smoothing",
        "augment_rotate", "augment_mirror", "augment_speed", "augment_noise_sigma",
        "seq_cutout_p", "seq_cutout_segments", "seq_cutout_ratio",
        "body_scale_jitter_lo", "body_scale_jitter_hi",
        "mixup_alpha", "cutmix_alpha",
        "use_center_loss", "lambda_center", "lambda_tau", "center_lr",
    }
    model_kwargs = {
        k: v for k, v in cfg_dict.items()
        if k not in skip_keys
    }
    # Normalise "model_name" → "name"
    if "model_name" in model_kwargs:
        model_kwargs["name"] = model_kwargs.pop("model_name")
    model = SimpleNamespace(**model_kwargs)
    # Compatibility: some from_config methods use config.model.clip_length
    if not hasattr(model, "clip_length"):
        model.clip_length = cfg_dict.get("clip_length", 64)
    return SimpleNamespace(model=model, skeleton=skeleton)


def _load_checkpoint_into_model(ckpt_path: Path, model: torch.nn.Module) -> None:
    """Load Lightning checkpoint state_dict into ``model`` (strip ``model.``)."""
    c = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = c["state_dict"]
    prefix = "model."
    stripped = {
        k[len(prefix):]: v for k, v in sd.items()
        if k.startswith(prefix)
    }
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if unexpected:
        click.echo(f"  [warn] unexpected keys: {unexpected[:3]}...", err=True)
    # Missing keys can happen for e.g. ArcMargin heads in the Lightning
    # module that aren't part of the pure model. Warn but don't fail.


@torch.no_grad()
def _infer_test(model: torch.nn.Module, clips: list, target_repr: str,
                clip_length: int, device: torch.device) -> np.ndarray:
    """Run inference on all test clips, return (N, num_class) logits."""
    model = model.to(device).eval()
    from diema.data.dataset import MotionDataset
    from torch.utils.data import DataLoader

    # Build a test MotionDataset. We load the test NPZ directly (instead of
    # via DataModule) because DataModule assumes a split dict.
    from utils.env import EnvConfig
    env = EnvConfig()
    test_npz = env.processed_dir / "motion_test_quat.npz"
    ds = MotionDataset(
        data_path=str(test_npz),
        indices=None,
        clip_length=clip_length,
        target_repr=target_repr,
        is_test=True,
        augmentation_pipeline=None,
    )
    loader = DataLoader(ds, batch_size=64, num_workers=4, shuffle=False)

    all_logits = []
    for batch in loader:
        x, _labels, _fnames = batch
        x = x.to(device)
        out = model(x)
        logits = out["logits"] if isinstance(out, dict) else out
        all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0), ds.filenames


@click.command()
@click.option("--experiments", required=True, multiple=True,
              help="Experiment folder names (e.g. exp002_a04_smooth).")
@click.option("--num-folds", default=10, type=int)
@click.option("--output-dir", default="output/predictions/ensemble_7way",
              help="Where to save per-exp test logits and aggregated logits.")
@click.option("--submission", default="output/submissions/ensemble_7way.csv",
              help="Path to save the final submission CSV.")
@click.option("--device", default="cuda",
              type=click.Choice(["cuda", "cpu"]))
def main(experiments: tuple, num_folds: int, output_dir: str,
         submission: str, device: str):
    env = EnvConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    click.echo(f"Inference device: {dev}")

    # Accumulate logits per experiment then across experiments
    exp_mean_logits = []
    ref_filenames = None
    for exp in experiments:
        exp_dir = env.artifacts_dir / exp
        fold_ckpts = sorted(exp_dir.glob("fold_*/best.ckpt"))
        fold_ckpts = [p for p in fold_ckpts
                      if p.parent.name.startswith("fold_") and
                      len(p.parent.name) == len("fold_") + 2]
        fold_ckpts = fold_ckpts[:num_folds]
        if not fold_ckpts:
            click.echo(f"[skip] no checkpoints for {exp}", err=True)
            continue
        click.echo(f"\n=== {exp}: {len(fold_ckpts)} folds ===")

        per_fold_logits = []
        for ckpt_path in fold_ckpts:
            fold_name = ckpt_path.parent.name
            cfg_path = ckpt_path.parent / "config.yaml"
            cfg_dict = OmegaConf.to_container(
                OmegaConf.load(str(cfg_path)), resolve=True,
            )
            builder_ns = _cfg_to_builder_ns(cfg_dict)
            model = build_model(builder_ns)
            _load_checkpoint_into_model(ckpt_path, model)
            logits, fnames = _infer_test(
                model,
                clips=None,
                target_repr=cfg_dict.get("target_repr", "6d"),
                clip_length=cfg_dict.get("clip_length", 64),
                device=dev,
            )
            if ref_filenames is None:
                ref_filenames = list(fnames)
            else:
                if list(fnames) != ref_filenames:
                    raise RuntimeError(
                        f"filename order mismatch in {exp}/{fold_name}"
                    )
            per_fold_logits.append(logits)
            click.echo(
                f"  {fold_name}: logits {logits.shape}, "
                f"argmax class distribution {np.bincount(logits.argmax(1), minlength=12).tolist()}"
            )

        # Mean across folds for this experiment
        mean_logits = np.stack(per_fold_logits, axis=0).mean(axis=0)
        np.save(out / f"{exp}_test_logits.npy", mean_logits)
        click.echo(f"  saved: {out / (exp + '_test_logits.npy')}  shape={mean_logits.shape}")
        exp_mean_logits.append(mean_logits)

    # Equal-weight average across experiments (decision D6)
    A = np.stack(exp_mean_logits, axis=0)
    ensemble = A.mean(axis=0)
    np.save(out / "ensemble_logits.npy", ensemble)
    click.echo(f"\nEnsemble logits saved: {out / 'ensemble_logits.npy'}  shape={ensemble.shape}")

    # Build submission
    preds = ensemble.argmax(axis=1)
    emotions = [IDX_TO_EMOTION[int(p)] for p in preds]
    sub = pd.DataFrame({
        "sample_name": ref_filenames,
        "predicted_label": preds,
        "predicted_emotion": emotions,
    })
    # Probability columns (softmax)
    probs = np.exp(ensemble - ensemble.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    for c in range(12):
        sub[f"prob_{IDX_TO_EMOTION[c]}"] = probs[:, c]

    sub_path = Path(submission)
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(sub_path, index=False)
    click.echo(f"Submission written: {sub_path} ({len(sub)} rows)")

    # Class distribution sanity
    dist = np.bincount(preds, minlength=12)
    click.echo("Predicted class distribution:")
    for c in range(12):
        click.echo(f"  {IDX_TO_EMOTION[c]:<12}: {dist[c]:>4} ({dist[c] / len(preds) * 100:.1f}%)")


if __name__ == "__main__":
    main()
