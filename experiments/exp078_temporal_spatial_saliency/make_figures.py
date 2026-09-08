"""exp078 — paper-figure rendering from saliency results.npz.

Produces:
  output/figures/exp078_temporal_saliency_per_class.png
  output/figures/exp078_spatial_saliency_per_class.png
  output/figures/exp078_joint_time_heatmap_per_class.png
  output/figures/exp078_top_features_table.md

Designed for Best Explainability Award submission. Each panel has a
clear annotation + colorbar; emotion labels canonical from
diema.data.parser.IDX_TO_EMOTION.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Per-row min-max normalize so each emotion's salience is in [0, 1]."""
    out = arr.astype(np.float64)
    for i in range(out.shape[0]):
        row = out[i]
        rmin, rmax = row.min(), row.max()
        out[i] = (row - rmin) / (rmax - rmin + 1e-9)
    return out


def plot_temporal_saliency(temporal_sal: np.ndarray, emotion_labels: list[str], out_path: str, normalize: str = "rows") -> None:
    """`normalize='rows'`: per-row min-max (visualizes within-emotion temporal pattern).
    `normalize='global'`: shared min-max (cross-emotion magnitude comparable)."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
    if normalize == "rows":
        ts = normalize_rows(temporal_sal)
        suffix = "row min-max"
    elif normalize == "global":
        ts = (temporal_sal - temporal_sal.min()) / (temporal_sal.max() - temporal_sal.min() + 1e-9)
        suffix = "global min-max"
    else:
        ts = temporal_sal
        suffix = "raw"
    im = ax.imshow(ts, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(12))
    ax.set_yticklabels(emotion_labels)
    ax.set_xlabel("frame (out of 64)")
    ax.set_title(f"Temporal saliency per emotion (gradient × input, exp034 a00, 10-fold LPO val, {suffix})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"normalized saliency ({suffix})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_saliency(spatial_sal: np.ndarray, emotion_labels: list[str], joint_names: list[str], out_path: str, normalize: str = "rows") -> None:
    """`rows`: within-emotion ranking; `global`: cross-emotion magnitude."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
    if normalize == "rows":
        ss = normalize_rows(spatial_sal)
        suffix = "row min-max"
    elif normalize == "global":
        ss = (spatial_sal - spatial_sal.min()) / (spatial_sal.max() - spatial_sal.min() + 1e-9)
        suffix = "global min-max"
    else:
        ss = spatial_sal
        suffix = "raw"
    im = ax.imshow(ss, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_yticks(range(12))
    ax.set_yticklabels(emotion_labels)
    ax.set_xticks(range(len(joint_names)))
    ax.set_xticklabels(joint_names, rotation=80, ha="right", fontsize=7)
    ax.set_title(f"Spatial saliency per emotion (CTV-25 joints, exp034 a00 10-fold LPO val, {suffix})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"normalized saliency ({suffix})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_joint_time_heatmap(joint_time_sal: np.ndarray, emotion_labels: list[str], joint_names: list[str], out_path: str) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(20, 11))
    vmin, vmax = joint_time_sal.min(), joint_time_sal.max()
    for c in range(12):
        ax = axes[c // 4, c % 4]
        im = ax.imshow(joint_time_sal[c].T, aspect="auto", cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(emotion_labels[c], fontsize=11)
        if c % 4 == 0:
            ax.set_ylabel("joint")
        if c // 4 == 2:
            ax.set_xlabel("frame")
        ax.set_yticks(range(0, len(joint_names), 4))
        ax.set_yticklabels([joint_names[i] for i in range(0, len(joint_names), 4)], fontsize=7)
    fig.suptitle("Joint × time saliency per emotion (CTV-25 joints × T=64 frames, exp034 a00)", fontsize=14)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, location="right")
    cbar.set_label("raw saliency (|grad × input|)")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_distinctive_features_table(temporal_sal: np.ndarray, spatial_sal: np.ndarray, emotion_labels: list[str], joint_names: list[str], out_path: str, k: int = 3) -> None:
    """Find features where one class has unusually high saliency relative
    to the average across emotions. Highlights emotion-distinctive cues
    not shared by all classes (Head/Neck dominate everywhere, so the
    top-k absolute table is dominated by those — this one shows what's
    *specific* to each emotion).

    Computes Δ on **unnormalized** saliency means (a follow-up review
    Issue 3 fix): subtract per-feature across-emotion mean from raw
    per-emotion saliency, so "distinctive" measures genuine class
    specificity without row-normalization artifacts."""
    spatial_mean_across_emotion = spatial_sal.mean(axis=0)
    temporal_mean_across_emotion = temporal_sal.mean(axis=0)
    spatial_distinctive = spatial_sal - spatial_mean_across_emotion[None, :]
    temporal_distinctive = temporal_sal - temporal_mean_across_emotion[None, :]
    lines = ["# exp078 — emotion-distinctive saliency features",
             "",
             "Per-emotion features where saliency is unusually high relative to the",
             "across-emotion average. Highlights emotion-specific cues not shared",
             "by all classes. Δ = (this class) − (avg across all 12 classes), in the",
             "[0, 1] normalized scale where each row's max is 1.0.",
             "",
             "| emotion | distinctive frames (idx, Δ) | distinctive joints (name, Δ) |",
             "|---|---|---|"]
    for c in range(12):
        t_idx = np.argsort(-temporal_distinctive[c])[:k]
        t_vals = temporal_distinctive[c][t_idx]
        s_idx = np.argsort(-spatial_distinctive[c])[:k]
        s_vals = spatial_distinctive[c][s_idx]
        t_str = ", ".join(f"f{ti} ({tv:+.2f})" for ti, tv in zip(t_idx, t_vals))
        s_str = ", ".join(f"{joint_names[si]} ({sv:+.2f})" for si, sv in zip(s_idx, s_vals))
        lines.append(f"| {emotion_labels[c]} | {t_str} | {s_str} |")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def make_top_features_table(temporal_sal: np.ndarray, spatial_sal: np.ndarray, emotion_labels: list[str], joint_names: list[str], out_path: str, k: int = 3) -> None:
    lines = ["# exp078 — top kinematic features per emotion class",
             "",
             "Top-k frames (T=64) and joints (V=25) by normalized gradient × input saliency,",
             "averaged over 10-fold LPO val (model: exp034 a00, Region-Aware Conv1D+Tr).",
             "",
             "| emotion | top-3 frames (index, salience) | top-3 joints (name, salience) |",
             "|---|---|---|"]
    for c in range(12):
        t_idx = np.argsort(-temporal_sal[c])[:k]
        t_vals = temporal_sal[c][t_idx]
        t_norm = t_vals / t_vals.max() if t_vals.max() > 0 else t_vals
        s_idx = np.argsort(-spatial_sal[c])[:k]
        s_vals = spatial_sal[c][s_idx]
        s_norm = s_vals / s_vals.max() if s_vals.max() > 0 else s_vals
        t_str = ", ".join(f"f{ti} ({tv:.2f})" for ti, tv in zip(t_idx, t_norm))
        s_str = ", ".join(f"{joint_names[si]} ({sv:.2f})" for si, sv in zip(s_idx, s_norm))
        lines.append(f"| {emotion_labels[c]} | {t_str} | {s_str} |")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-npz", default="experiments/exp078_temporal_spatial_saliency/results.npz")
    parser.add_argument("--output-dir", default="output/figures")
    args = parser.parse_args()

    data = np.load(args.results_npz, allow_pickle=True)
    temporal_sal = data["temporal_sal"]
    spatial_sal = data["spatial_sal"]
    joint_time_sal = data["joint_time_sal"]
    emotion_labels = list(data["emotion_labels"])
    joint_names = list(data["joint_names"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_temporal_saliency(temporal_sal, emotion_labels, str(out_dir / "exp078_temporal_saliency_per_class.png"), normalize="rows")
    plot_temporal_saliency(temporal_sal, emotion_labels, str(out_dir / "exp078_temporal_saliency_global.png"), normalize="global")
    plot_spatial_saliency(spatial_sal, emotion_labels, joint_names, str(out_dir / "exp078_spatial_saliency_per_class.png"), normalize="rows")
    plot_spatial_saliency(spatial_sal, emotion_labels, joint_names, str(out_dir / "exp078_spatial_saliency_global.png"), normalize="global")
    plot_joint_time_heatmap(joint_time_sal, emotion_labels, joint_names, str(out_dir / "exp078_joint_time_heatmap_per_class.png"))
    make_top_features_table(temporal_sal, spatial_sal, emotion_labels, joint_names, str(out_dir / "exp078_top_features_table.md"))
    make_distinctive_features_table(temporal_sal, spatial_sal, emotion_labels, joint_names, str(out_dir / "exp078_distinctive_features_table.md"))
    print(f"[exp078] saved 7 artifacts under {out_dir}/")


if __name__ == "__main__":
    main()
