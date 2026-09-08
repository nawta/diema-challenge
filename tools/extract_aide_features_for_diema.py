"""exp077 — extract AIDE-trained UbH-GCN penultimate features for DIEM-A clips.

For each DIEM-A clip in motion_{train,test}_quat.npz:
  1. Project BVH-24 → AIDE Halpe-14 layout via diema.features.bvh_to_halpe14
  2. Apply joint→bone transform for the 2 bone streams
  3. Forward through 4 frozen UbH-GCN streams (joint_root_1, joint_root_14,
     bone_root_1, bone_root_14)
  4. Capture penultimate (256-d) features via forward hook on `self.fc`
  5. Concat 4 × 256 → 1024-d feature per clip
  6. Save to data/diema_challenge/processed/aide_features_b00.npz

This is the data-prep step for exp077 linear-probe smoke gate and
conditional full B2 concat.

Usage:
    conda run -n acii2026 python tools/extract_aide_features_for_diema.py \
        --motion-npz data/diema_challenge/processed/motion_train_quat.npz \
        --output data/diema_challenge/processed/aide_features_b00_train.npz \
        --device cuda:0

See: experiments/exp077_aide_feature_transfer
Related: tmp/UbH-GCN-source/model/UbHGCN.py (the 4 stream model)
         tmp/UbH-GCN-source/feeders/feeder_aide.py (joint→bone transform reference)
         tmp/UbH-GCN-source/work_dir/{joint,bone}_root_{1,14}/model_best.pt (trained ckpts)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import click
import numpy as np
import torch

# Make UbH-GCN repo importable for `model.UbHGCN.Model` / `graph.aide_hierarchy.Graph`.
UBH_GCN_SRC = Path(__file__).resolve().parent.parent / "tmp" / "UbH-GCN-source"
if str(UBH_GCN_SRC) not in sys.path:
    sys.path.insert(0, str(UBH_GCN_SRC))

from diema.features.bvh_to_halpe14 import project_bvh_to_halpe14

# AIDE bone pairs (1-indexed in source). The bone modality replaces
# data[v1-1] ← data[v1-1] - data[v2-1] for each pair.
AIDE_BONE_PAIRS_UPPER_HEAD = (
    (1, 13), (2, 1), (3, 1), (4, 2), (5, 3), (6, 13),
    (7, 13), (8, 6), (9, 7), (10, 8), (11, 9), (12, 1), (13, 14), (14, 13),
)


def joint_to_bone(data: np.ndarray) -> np.ndarray:
    """Apply AIDE bone transform. Input/output shape (3, T, 14)."""
    bone = np.zeros_like(data)
    for v1, v2 in AIDE_BONE_PAIRS_UPPER_HEAD:
        bone[:, :, v1 - 1] = data[:, :, v1 - 1] - data[:, :, v2 - 1]
    return bone


def load_ubh_gcn_stream(checkpoint_path: Path, root: int, device: torch.device) -> torch.nn.Module:
    """Load a UbH-GCN-Model with weights from `model_best.pt`.

    `root` ∈ {1, 14} selects the hierarchical-graph root (the AIDE config
    field `graph_args.root`).
    """
    from model.UbHGCN import Model  # noqa: E402

    model = Model(
        num_class=5,
        num_point=14,
        num_person=1,
        graph="graph.aide_hierarchy.Graph",
        graph_args={"labeling_mode": "spatial", "root": root},
        in_channels=3,
        drop_out=0.25,
        adaptive=True,
        base_channels=64,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state and not any(k.startswith("data_bn") for k in state):
            state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def attach_penultimate_hook(model: torch.nn.Module) -> list[torch.Tensor]:
    """Hook the `fc` layer; on each forward pass append the input (256-d) to a list."""
    sink: list[torch.Tensor] = []

    def _hook(_module, inputs, _output):
        # inputs is a tuple of length 1: (N, 256)
        sink.append(inputs[0].detach().cpu())

    model.fc.register_forward_hook(_hook)
    return sink


@click.command()
@click.option("--motion-npz", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--output", type=click.Path(dir_okay=False), required=True)
@click.option("--checkpoint-root", type=click.Path(exists=True, file_okay=False),
              default=str(UBH_GCN_SRC / "work_dir"),
              help="Folder containing {joint,bone}_root_{1,14}/model_best.pt")
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--batch-size", default=64, show_default=True, type=int)
@click.option("--target-frames", default=16, show_default=True, type=int)
@click.option("--normalize", default="per_clip", show_default=True,
              type=click.Choice(["per_clip", "dataset_affine", "none"]),
              help="how to map projected coords into AIDE's pixel range")
@click.option("--drop-axis", default="y", show_default=True,
              type=click.Choice(["x", "y", "z"]),
              help="which BVH axis to drop for orthographic projection")
def main(motion_npz: str, output: str, checkpoint_root: str, device: str, batch_size: int, target_frames: int, normalize: str, drop_axis: str) -> None:
    """Extract 4 × 256-d AIDE features for every clip in motion_npz."""
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    npz = np.load(motion_npz, allow_pickle=True)
    num_clips = int(npz["num_clips"])
    filenames = list(npz["filenames"])
    print(f"[extract] {num_clips} clips from {motion_npz}", flush=True)

    streams = {
        "joint_root_1":  load_ubh_gcn_stream(Path(checkpoint_root) / "joint_root_1"  / "model_best.pt", root=1,  device=torch_device),
        "joint_root_14": load_ubh_gcn_stream(Path(checkpoint_root) / "joint_root_14" / "model_best.pt", root=14, device=torch_device),
        "bone_root_1":   load_ubh_gcn_stream(Path(checkpoint_root) / "bone_root_1"   / "model_best.pt", root=1,  device=torch_device),
        "bone_root_14":  load_ubh_gcn_stream(Path(checkpoint_root) / "bone_root_14"  / "model_best.pt", root=14, device=torch_device),
    }
    hooks = {name: attach_penultimate_hook(m) for name, m in streams.items()}

    features = {name: np.zeros((num_clips, 256), dtype=np.float32) for name in streams}
    logits = {name: np.zeros((num_clips, 5), dtype=np.float32) for name in streams}

    t0 = time.time()
    for start in range(0, num_clips, batch_size):
        end = min(start + batch_size, num_clips)
        batch_idx = range(start, end)
        joint_batch = np.zeros((end - start, 3, target_frames, 14, 1), dtype=np.float32)
        for j, i in enumerate(batch_idx):
            rp = np.asarray(npz[f"clip_{i}_root_pos"], dtype=np.float32)
            jq = np.asarray(npz[f"clip_{i}_joint_data"], dtype=np.float32)
            joint_batch[j, :, :, :, 0] = project_bvh_to_halpe14(
                rp, jq, target_frames=target_frames,
                drop_axis=drop_axis, normalize=normalize,
            )

        bone_batch = joint_batch.copy()
        for j in range(bone_batch.shape[0]):
            bone_batch[j, :, :, :, 0] = joint_to_bone(bone_batch[j, :, :, :, 0])

        joint_t = torch.from_numpy(joint_batch).to(torch_device)
        bone_t = torch.from_numpy(bone_batch).to(torch_device)

        for name in streams:
            hooks[name].clear()

        with torch.no_grad():
            for name in ("joint_root_1", "joint_root_14"):
                logit = streams[name](joint_t)
                logits[name][start:end] = logit.cpu().numpy()
            for name in ("bone_root_1", "bone_root_14"):
                logit = streams[name](bone_t)
                logits[name][start:end] = logit.cpu().numpy()

        for name in streams:
            features[name][start:end] = hooks[name][0].numpy()

        if (start // batch_size) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{end}/{num_clips}] {elapsed:.1f}s elapsed", flush=True)

    concat = np.concatenate([features[n] for n in ("joint_root_1", "joint_root_14", "bone_root_1", "bone_root_14")], axis=1)
    print(f"[extract] concat shape: {concat.shape} (expected (N, 1024))")
    print(f"[extract] mean variance per dim: {concat.var(axis=0).mean():.4f}")
    dead = (concat.std(axis=0) < 1e-4).sum()
    print(f"[extract] dead features (std<1e-4): {dead}/1024")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.savez(
        output,
        filenames=filenames,
        features_concat=concat,
        features_joint_root_1=features["joint_root_1"],
        features_joint_root_14=features["joint_root_14"],
        features_bone_root_1=features["bone_root_1"],
        features_bone_root_14=features["bone_root_14"],
        logits_joint_root_1=logits["joint_root_1"],
        logits_joint_root_14=logits["joint_root_14"],
        logits_bone_root_1=logits["bone_root_1"],
        logits_bone_root_14=logits["bone_root_14"],
    )
    print(f"[extract] saved {output} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
