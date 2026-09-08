"""exp079 — classical kinematic features per emotion class.

Computes traditional motion-analysis features per emotion class on DIEM-A
train clips and identifies the most-distinguishing features per emotion.
Bridges the deep-saliency narrative (exp078) to classical kinematics
that body-language researchers (the likely paper audience) interpret.

Features per joint:
  - mean joint speed  = mean ||v_t||₂ over t
  - peak joint speed  = max ||v_t||₂ over t
  - mean acceleration = mean ||a_t||₂ over t
  - range of motion   = max||p_t − p̄||₂ over t (radius)
  - motion energy     = ∑_t ||v_t||₂²

All computed in 3D world coords (FK on BVH quaternions).

Output:
  results.npz with (12 emotions, 24 BVH joints, 5 features) tensor
  output/figures/exp079_kinematic_features_per_emotion.png
  output/figures/exp079_distinctive_kinematics_table.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diema.data.parser import IDX_TO_EMOTION, emotion_to_label
from diema.features.multi_stream import compute_joint_positions
from diema.models.skeleton_graph import DIEMA_JOINT_NAMES

BVH_JOINT_NAMES = [
    "Hips", "Spine", "Spine1", "Spine2", "Spine3",
    "Neck", "Neck1", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
]
FEATURE_NAMES = ["mean_speed", "peak_speed", "mean_accel", "range_of_motion", "motion_energy"]


def per_clip_features(root_pos: np.ndarray, joint_quats: np.ndarray) -> np.ndarray:
    """Return (24, 5) feature matrix for one clip."""
    world_pos = compute_joint_positions(root_pos, joint_quats)  # (F, 24, 3)
    velocity = np.diff(world_pos, axis=0)                       # (F-1, 24, 3)
    accel = np.diff(velocity, axis=0)                            # (F-2, 24, 3)
    speed = np.linalg.norm(velocity, axis=-1)                    # (F-1, 24)
    accel_mag = np.linalg.norm(accel, axis=-1)                   # (F-2, 24)
    pos_centroid = world_pos.mean(axis=0, keepdims=True)         # (1, 24, 3)
    radii = np.linalg.norm(world_pos - pos_centroid, axis=-1)    # (F, 24)
    feats = np.stack([
        speed.mean(axis=0),       # mean speed
        speed.max(axis=0),        # peak speed
        accel_mag.mean(axis=0),   # mean acceleration
        radii.max(axis=0),        # range of motion
        (speed ** 2).sum(axis=0), # motion energy
    ], axis=-1)                    # (24, 5)
    return feats.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-npz", default="data/diema_challenge/processed/motion_train_quat.npz")
    parser.add_argument("--output-npz", default="experiments/exp079_classical_kinematic_features/results.npz")
    parser.add_argument("--output-figdir", default="output/figures")
    args = parser.parse_args()

    t0 = time.time()
    npz = np.load(args.motion_npz, allow_pickle=True)
    n_clips = int(npz["num_clips"])
    print(f"[exp079] processing {n_clips} clips", flush=True)

    feat_sum = np.zeros((12, 24, 5), dtype=np.float64)
    feat_sq_sum = np.zeros((12, 24, 5), dtype=np.float64)
    count_per_class = np.zeros(12, dtype=np.int64)
    for i in range(n_clips):
        fn = str(npz["filenames"][i])
        try:
            label = emotion_to_label(fn)
        except (ValueError, KeyError):
            continue
        rp = np.asarray(npz[f"clip_{i}_root_pos"], dtype=np.float32)
        jq = np.asarray(npz[f"clip_{i}_joint_data"], dtype=np.float32)
        feats = per_clip_features(rp, jq)
        feat_sum[label] += feats
        feat_sq_sum[label] += feats ** 2
        count_per_class[label] += 1
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{n_clips}] {time.time() - t0:.1f}s", flush=True)

    feat_mean = feat_sum / count_per_class[:, None, None].clip(min=1)
    feat_std = np.sqrt(feat_sq_sum / count_per_class[:, None, None].clip(min=1) - feat_mean ** 2 + 1e-9)

    feat_log_mean = np.log1p(feat_mean)
    feat_z = (feat_log_mean - feat_log_mean.mean(axis=0)[None, :, :]) / (feat_log_mean.std(axis=0)[None, :, :] + 1e-9)

    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        feat_mean=feat_mean.astype(np.float32),
        feat_std=feat_std.astype(np.float32),
        feat_log_mean=feat_log_mean.astype(np.float32),
        feat_zscore_per_feature=feat_z.astype(np.float32),
        count_per_class=count_per_class,
        bvh_joint_names=np.array(BVH_JOINT_NAMES),
        feature_names=np.array(FEATURE_NAMES),
        emotion_labels=np.array([IDX_TO_EMOTION[c] for c in range(12)]),
    )

    figdir = Path(args.output_figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(22, 6))
    for fi, fname in enumerate(FEATURE_NAMES):
        ax = axes[fi]
        im = ax.imshow(feat_z[:, :, fi], aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_title(fname, fontsize=11)
        ax.set_yticks(range(12))
        ax.set_yticklabels([IDX_TO_EMOTION[c] for c in range(12)], fontsize=8)
        ax.set_xticks(range(24))
        ax.set_xticklabels(BVH_JOINT_NAMES, rotation=80, ha="right", fontsize=6)
    fig.suptitle("Classical kinematic features per emotion (z-score over emotion axis per (joint, feature))", fontsize=13)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, location="right", label="z-score")
    fig.savefig(figdir / "exp079_kinematic_features_per_emotion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = ["# exp079 — emotion-distinctive classical kinematic features",
             "",
             "Z-score of each (joint, feature) within an emotion class against the",
             "across-emotion distribution. Positive Δ = emotion uses that joint's",
             "feature MORE than average; negative Δ = LESS than average.",
             "All features computed on FK-projected 3D BVH world positions over all",
             "DIEM-A train clips (n=7992). Top-3 distinctive (joint, feature) pairs",
             "per emotion shown.",
             "",
             "| emotion | top-1 (joint, feature, z) | top-2 | top-3 |",
             "|---|---|---|---|"]
    for c in range(12):
        flat = feat_z[c].reshape(-1)
        top = np.argsort(-flat)[:3]
        entries = []
        for idx in top:
            j_idx, f_idx = idx // 5, idx % 5
            entries.append(f"({BVH_JOINT_NAMES[j_idx]}, {FEATURE_NAMES[f_idx]}, {feat_z[c, j_idx, f_idx]:+.2f})")
        lines.append(f"| {IDX_TO_EMOTION[c]} | {entries[0]} | {entries[1]} | {entries[2]} |")
    with open(figdir / "exp079_distinctive_kinematics_table.md", "w") as f:
        f.write("\n".join(lines))

    print(f"[exp079] saved {args.output_npz} + 2 figures in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
