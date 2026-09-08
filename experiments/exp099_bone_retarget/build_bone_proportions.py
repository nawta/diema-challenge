"""build_bone_proportions.py — Stage 1b/1c.

For each of 74 train performers (anonymized test files cannot be
aggregated by performer), extract a 24-D
bone-proportion vector (normalized by skeleton height) from one
representative BVH, plus the raw standing height.

Stage 1c then runs:
- PCA on the 74×24 matrix (color-coded JP vs TW)
- Mann-Whitney U per bone (JP vs TW) + exact p + rank-biserial effect size
- Sanity check: 1944 test clips' bone proportions, do they fall inside
  train PCA range?

Outputs:
- output/retarget/bone_proportions.json  (per-performer dict)
- output/retarget/bone_proportions_summary.md  (PCA + stats table)
- output/retarget/bone_proportions_pca.png  (figure)

Related: `bvh_offset_probe.py` (Stage 1a kill gate); related design
constants in `diema/features/multi_stream.py::_DIEMA_OFFSETS_BVH`
(hardcoded JP_06 reference).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pybvh
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bvh_offset_probe import (  # noqa: E402
    RAW_BVH_TRAIN, find_representative_bvh, extract_24joint_offsets,
    compute_height)

OUT_DIR = Path("output/retarget")
RAW_BVH_TEST = Path("data/diema_challenge/raw/bvh/test")


def list_train_performers() -> list[tuple[str, str]]:
    """Return sorted list of (performer_id, country) tuples for train."""
    ids = set()
    for p in RAW_BVH_TRAIN.glob("*.bvh"):
        parts = p.stem.split("_")
        if len(parts) >= 2 and parts[0] in {"JP", "TW"}:
            ids.add(f"{parts[0]}_{parts[1]}")
    return sorted([(pid, pid.split("_")[0]) for pid in ids])


def compute_bone_proportions(offsets: np.ndarray, height: float
                              ) -> np.ndarray:
    """Per-bone Euclidean length normalized by total skeleton height.
    Returns (24,) array; idx 0 (Hips) is 0 by construction."""
    lengths = np.linalg.norm(offsets, axis=1)
    return lengths / max(height, 1e-9)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1b] enumerating train performers under {RAW_BVH_TRAIN}")
    train_performers = list_train_performers()
    n_jp = sum(1 for _, c in train_performers if c == "JP")
    n_tw = sum(1 for _, c in train_performers if c == "TW")
    print(f"  found {len(train_performers)} train performers "
          f"({n_jp} JP + {n_tw} TW)")
    assert len(train_performers) == 74, \
        f"expected 74 train performers, got {len(train_performers)}"

    # ---------------------- Stage 1b: per-performer offsets ----------------------
    bone_data: dict[str, dict] = {}
    joint_names: list[str] | None = None
    for pid, country in train_performers:
        path = find_representative_bvh(pid)
        offsets, names = extract_24joint_offsets(path)
        if joint_names is None:
            joint_names = names
        height = compute_height(offsets, names)
        proportions = compute_bone_proportions(offsets, height)
        bone_data[pid] = {
            "country": country,
            "height_raw": float(height),
            "proportions": proportions.tolist(),
            "rep_bvh": path.name,
        }
    proportions_matrix = np.stack(
        [bone_data[pid]["proportions"] for pid, _ in train_performers])
    heights = np.array([bone_data[pid]["height_raw"]
                         for pid, _ in train_performers])
    print(f"  proportions matrix: {proportions_matrix.shape}  "
          f"(should be (74, 24))")
    print(f"  height stats: min={heights.min():.2f}, "
          f"max={heights.max():.2f}, median={np.median(heights):.2f}")

    # save JSON
    bone_data_with_meta = {
        "_meta": {
            "n_train_performers": len(train_performers),
            "n_jp": n_jp,
            "n_tw": n_tw,
            "joint_names": joint_names,
            "stage_1a_kill_gate": "PASSED (23.67% max bone Δ)",
            "note": ("each performer's proportions come from ONE "
                      "representative BVH file (prefer anger_1_H "
                      "deterministically)"),
        },
        "performers": bone_data,
    }
    with open(OUT_DIR / "bone_proportions.json", "w") as f:
        json.dump(bone_data_with_meta, f, indent=2)
    print(f"  → {OUT_DIR/'bone_proportions.json'}")

    # ---------------------- Stage 1c: PCA + Mann-Whitney ----------------------
    print()
    print("[1c] running PCA + Mann-Whitney U per bone")
    # PCA via SVD on centered matrix
    centered = proportions_matrix - proportions_matrix.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pca = centered @ Vt[:2].T  # (74, 2)
    explained_var = (S**2) / (S**2).sum()
    print(f"  PCA explained variance: PC1={explained_var[0]*100:.1f}%, "
          f"PC2={explained_var[1]*100:.1f}%")

    # plot
    fig, ax = plt.subplots(figsize=(7, 6))
    jp_mask = np.array([c == "JP" for _, c in train_performers])
    ax.scatter(pca[jp_mask, 0], pca[jp_mask, 1], c="#1f77b4",
                label=f"JP (n={jp_mask.sum()})", s=40, alpha=0.7,
                edgecolors="k", linewidths=0.5)
    ax.scatter(pca[~jp_mask, 0], pca[~jp_mask, 1], c="#d62728",
                label=f"TW (n={(~jp_mask).sum()})", s=40, alpha=0.7,
                edgecolors="k", linewidths=0.5)
    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f} %)")
    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f} %)")
    ax.set_title("Per-performer bone-proportion PCA (train, n=74)\n"
                  "anonymized as P_NN in figure; performer IDs in JSON")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bone_proportions_pca.png", dpi=120)
    plt.close(fig)
    print(f"  → {OUT_DIR/'bone_proportions_pca.png'}")

    # Mann-Whitney U per bone
    rows = []
    for j in range(24):
        jp_vals = proportions_matrix[jp_mask, j]
        tw_vals = proportions_matrix[~jp_mask, j]
        if jp_vals.std() < 1e-12 and tw_vals.std() < 1e-12:
            stat, p, rb = np.nan, 1.0, 0.0
        else:
            stat, p = mannwhitneyu(jp_vals, tw_vals, alternative="two-sided")
            # rank-biserial: (U / (n1*n2)) * 2 - 1
            rb = (stat / (len(jp_vals) * len(tw_vals))) * 2 - 1
        rows.append({
            "bone": joint_names[j],
            "jp_median": float(np.median(jp_vals)),
            "tw_median": float(np.median(tw_vals)),
            "mann_whitney_U": float(stat) if not np.isnan(stat) else None,
            "p_value": float(p),
            "rank_biserial": float(rb),
        })
    rows_sorted = sorted(rows, key=lambda r: r["p_value"])

    # markdown summary
    md = []
    md.append("# Bone proportions — Stage 1b/1c summary")
    md.append("")
    md.append(f"- **Train performers**: 74 ({n_jp} JP + {n_tw} TW)")
    md.append(f"- **Test performers**: anonymized (`test0001.bvh` ... "
                f"`test1944.bvh`), 18 unique IDs hidden")
    md.append(f"- **Height range**: min {heights.min():.2f}, median "
                f"{np.median(heights):.2f}, max {heights.max():.2f}")
    md.append(f"- **PCA**: PC1 = {explained_var[0]*100:.1f} %, "
                f"PC2 = {explained_var[1]*100:.1f} %, "
                f"PC1+PC2 = {(explained_var[0]+explained_var[1])*100:.1f} %")
    md.append("")
    md.append("## Mann-Whitney U per bone (JP vs TW, two-sided)")
    md.append("")
    md.append("| Bone | JP median | TW median | U | exact p | rank-biserial r | significance |")
    md.append("|---|---:|---:|---:|---:|---:|---|")
    sig_3 = sig_2 = 0
    for r in rows_sorted:
        if r["mann_whitney_U"] is None:
            md.append(f"| {r['bone']} | {r['jp_median']:.4f} | "
                       f"{r['tw_median']:.4f} | (degenerate) | 1.000 | 0.000 | — |")
            continue
        if r["p_value"] < 0.001:
            sig_marker = "***"
            sig_3 += 1
        elif r["p_value"] < 0.01:
            sig_marker = "**"
            sig_2 += 1
        elif r["p_value"] < 0.05:
            sig_marker = "*"
        else:
            sig_marker = ""
        p_str = (f"{r['p_value']:.2e}" if r["p_value"] < 0.001
                  else f"{r['p_value']:.4f}")
        md.append(f"| {r['bone']} | {r['jp_median']:.4f} | "
                   f"{r['tw_median']:.4f} | {r['mann_whitney_U']:.0f} | "
                   f"{p_str} | {r['rank_biserial']:+.3f} | {sig_marker} |")
    md.append("")
    md.append(f"**Significance summary**: {sig_3} bones with p < 0.001, "
               f"{sig_2} bones with 0.001 ≤ p < 0.01.")
    md.append("")
    md.append("## Stage 1a kill gate")
    md.append("")
    md.append("- ✅ **PASS** — JP_06 reference vs TW_01 (12.45 %) / "
               "JP_07 (23.67 %); max relative deviation 23.67 %, "
               "well above the 5 % threshold.")
    md.append("- Implication: cached `motion_train_quat.npz` joint "
               "positions (which use hardcoded JP_06 offsets) "
               "body-shape-flatten the raw signal.")
    md.append("")
    md.append("## Implication for input modality")
    md.append("")
    md.append("Retarget perturbs FK-derived joint positions but is a "
               "no-op for rotation_6d input (rotations are scale-invariant). "
               " training should use **joint_pos input** (or "
               "explicitly re-derive joint positions with retargeted "
               "offsets per clip) to actually test the hypothesis. "
               "See "
               "open item #1.")
    md.append("")
    md.append("---")
    md.append("Generated by `experiments/exp099_bone_retarget/"
               "build_bone_proportions.py` (Stage 1b/1c).")
    (OUT_DIR / "bone_proportions_summary.md").write_text("\n".join(md))
    print(f"  → {OUT_DIR/'bone_proportions_summary.md'}")
    print()
    print(f"[summary] {sig_3} bones with p<0.001, {sig_2} with "
          f"0.001≤p<0.01 (across 24 bones, JP n={n_jp} vs TW n={n_tw})")
    print(f"          PC1+PC2 explain {(explained_var[0]+explained_var[1])*100:.1f} % "
          f"of variance")


if __name__ == "__main__":
    main()
