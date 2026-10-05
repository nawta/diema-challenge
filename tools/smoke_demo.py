"""Run the ensemble's skeleton models on random input. No data needed.

This script checks that the code installs and runs before you have the
DIEM-A data. It builds the 7 skeleton models that the submitted 11-model
ensemble trains from scratch, using each member's settings (``config.yaml``
plus the ``exp/<variant>.yaml`` used in the submission). It then feeds them
a random batch shaped like real model input and averages their raw logits
with equal weights, the logit-mean rule described in the README and the
paper.

The weights are random, so the predicted emotions are meaningless. The
script only shows the input/output shapes and that each model runs.

Input shape: (N, C=6, T=64, V=25), i.e. a batch of N clips, 6 channels per
joint (6D rotation), 64 frames, 25 joints.

The other 4 ensemble members (MotionBERT, C3D statistics, and two MAMP
probes) need pretrained weights or features made from the data, so they
are left out here.

Usage:
    python tools/smoke_demo.py
    python tools/smoke_demo.py --batch-size 4 --device cpu

Related: diema/models/registry.py (build_model), tests/test_new_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import click
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from diema.data.parser import IDX_TO_EMOTION  # noqa: E402
from diema.models import build_model  # noqa: E402
from diema.models.skeleton_graph import DIEMA_INWARD_EDGES  # noqa: E402
from utils.seed import seed_everything  # noqa: E402

# (display name, experiment directory, settings variant) for the from-scratch
# members of the submitted ensemble, as listed in
# experiments/exp088_final_ensemble_submission/build_final_submission.py.
MEMBERS = [
    ("ST-GCN++ (baseline)", "exp002_enhanced", "a04"),
    ("CTR-GCN", "exp003_ctrgcn", "a00"),
    ("SkateFormer", "exp004_skateformer", "a01"),
    ("ProtoGCN", "exp006_protogcn", "a00"),
    ("Conv1D+Transformer", "exp020_conv1d_transformer", "a01"),
    ("Keypoint-pool MLP", "exp023_keypoint_pool_mlp", "a00"),
    ("Region-Aware Conv-Transformer", "exp034_regionaware_convtr", "a00"),
]


def load_model_cfg(exp_dir: str, variant: str) -> SimpleNamespace:
    """Merge config.yaml with exp/<variant>.yaml and build the namespace build_model() expects."""
    exp_path = REPO_ROOT / "experiments" / exp_dir
    merged = OmegaConf.merge(
        OmegaConf.load(exp_path / "config.yaml"),
        OmegaConf.load(exp_path / "exp" / f"{variant}.yaml"),
    )
    raw = OmegaConf.to_container(merged)
    raw.pop("defaults", None)
    name = raw.pop("model_name")
    return SimpleNamespace(
        model=SimpleNamespace(name=name, **raw),
        skeleton=SimpleNamespace(num_nodes=raw.get("num_nodes", 25), inward_edges=DIEMA_INWARD_EDGES),
    )


@click.command()
@click.option("--batch-size", type=int, default=2, show_default=True)
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def main(batch_size: int, device: str, seed: int) -> None:
    seed_everything(seed)
    x = torch.randn(batch_size, 6, 64, 25, device=device)
    print(f"Random input batch: {tuple(x.shape)}  (N, channels, frames, joints)\n")

    all_logits = []
    print(f"{'model':32s} logits shape")
    for display, exp_dir, variant in MEMBERS:
        model = build_model(load_model_cfg(exp_dir, variant)).to(device).eval()
        with torch.no_grad():
            logits = model(x)["logits"]
        all_logits.append(logits)
        print(f"{display:32s} {tuple(logits.shape)}")

    fused = torch.stack(all_logits).mean(dim=0)
    pred = fused.argmax(dim=1).tolist()
    print(f"\nLogit-mean fusion of {len(all_logits)} models: {tuple(fused.shape)}")
    print("Predicted emotions (random weights, so these mean nothing):", [IDX_TO_EMOTION[i] for i in pred])


if __name__ == "__main__":
    main()
