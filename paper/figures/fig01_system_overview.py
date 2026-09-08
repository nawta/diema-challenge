"""paper/figures/fig01_system_overview.py — Fig.1 (P2).

Design doc: paper/data_inventory.md (P0 canonical numbers + open questions).
A SCHEMATIC (not data-driven): 1-column, three vertical bands —
(A) DIEM-A challenge structure, (B) 11-way ensemble pipeline, (C) outcomes.
Fully anonymized: no institution, author, first-person, or repo handle.

Verified facts (no fabrication; sources in P0 inventory):
  * train 74 perf JP40/TW34 = 7,992 clips  -> per_performer csv
  * 11-way logit-mean = 36.94% pooled / 36.80+/-4.00 per-fold;
    STGCN++ baseline 26.32 pooled / 25.73+/-4.03 (official-reported 25.2/27.1)
    -> paper/reconcile_f1.json
  * 92 perf / 9,935 clips total; test 18 perf / 1,944 clips
    (JP9/TW9 challenge spec; total JP49/TW43 consistent)
    -> docs/results_summary.md §caveat, challenge description
"""

from _style import (PALETTE_QUAL, ONE_COL, save_fig, seed_all, set_size,
                     setup_ieee)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

setup_ieee()
seed_all(42)


def _tint(hex_color: str, frac: float = 0.82) -> tuple:
    """Blend a palette color toward white for a light fill."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


def box(ax, x, y, w, h, text, *, fc, ec, fs=6.0, bold=False, lw=0.5,
        rounding=0.012):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={rounding}",
        linewidth=lw, edgecolor=ec, facecolor=fc, mutation_aspect=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=5)


def arrow(ax, x, y0, y1):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.5, color="0.35",
                                shrinkA=0, shrinkB=0))


fig, ax = plt.subplots()
set_size(fig, ONE_COL, 4.75)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# faint band separators + left-edge band tags (3-band clarity)
for ys in (0.792, 0.232):
    ax.plot([0.02, 0.98], [ys, ys], lw=0.3, ls=(0, (4, 3)), color="0.7",
            zorder=0)
for ty, tag in ((0.885, "Dataset"), (0.52, "Pipeline"), (0.13, "Outcomes")):
    ax.text(0.012, ty, tag, rotation=90, ha="center", va="center",
            fontsize=5.6, color="0.55", style="italic")

# ── Band A: dataset ───────────────────────────────────────────────────────
ax.text(0.5, 0.968, "DIEM-A Challenge", ha="center", va="center",
        fontsize=9.5, fontweight="bold")
box(ax, 0.05, 0.873, 0.91, 0.072,
    "Train  74 performers (JP 40 · TW 34)  →  7,992 clips\n"
    "Test   18 performers (JP 9 · TW 9)   →  1,944 clips",
    fc="#eef3f8", ec="0.5", fs=6.0)
box(ax, 0.05, 0.812, 0.91, 0.045,
    "12 emotions   |   3 scenarios × 3 intensities   |   "
    "BVH·FBX·C3D + text   |   performer-disjoint LPO",
    fc="#f6f5f0", ec="0.6", fs=5.3)

# ── Band B: pipeline ──────────────────────────────────────────────────────
box(ax, 0.28, 0.722, 0.44, 0.050,
    "25-joint skeleton · 64-frame clip", fc="white", ec="0.4", fs=6.2,
    bold=True)
arrow(ax, 0.5, 0.722, 0.700)

# ensemble container + 4 color-coded inductive-bias rows
box(ax, 0.05, 0.408, 0.91, 0.292, "", fc="#fbfbfb", ec="0.6", fs=6)
ax.text(0.5, 0.678, "11 models · 4 inductive biases", ha="center",
        va="center", fontsize=6.0, fontweight="bold")
rows = [
    ("GCN family", "STGCN++ · CTR-GCN · ProtoGCN", PALETTE_QUAL[0]),
    ("Attention", "SkateFormer", PALETTE_QUAL[1]),
    ("Hybrid / MLP", "Conv1D+Tr · Region-Aware · KP-MLP", PALETTE_QUAL[2]),
    ("External pretrain", "MotionBERT · C3D-stats · MAMP×2", PALETTE_QUAL[3]),
]
ry = [0.604, 0.548, 0.492, 0.436]
for (name, mem, col), y in zip(rows, ry):
    box(ax, 0.085, y, 0.83, 0.050, f"{name}  —  {mem}",
        fc=_tint(col), ec=col, fs=5.5, lw=0.7)
arrow(ax, 0.5, 0.408, 0.388)

box(ax, 0.18, 0.318, 0.64, 0.066,
    "Logit-mean fusion\nequal-weight 11-way", fc="#fff4e0", ec="#b06a00",
    fs=6.4, bold=True, lw=0.8)
arrow(ax, 0.5, 0.318, 0.292)
box(ax, 0.25, 0.236, 0.50, 0.052,
    "12-class softmax → emotion", fc="white", ec="0.4", fs=6.2,
    bold=True)

# ── Band C: outcomes ──────────────────────────────────────────────────────
box(ax, 0.05, 0.120, 0.43, 0.092,
    "Performance\nMacro-F1 36.94%\n(STGCN++ 25.2%)",
    fc=_tint(PALETTE_QUAL[2], 0.74), ec=PALETTE_QUAL[2], fs=5.8, bold=True)
box(ax, 0.52, 0.120, 0.43, 0.092,
    "Explainability\nkinematics · body parts\n· saliency maps",
    fc=_tint(PALETTE_QUAL[4], 0.74), ec=PALETTE_QUAL[4], fs=5.8, bold=True)

pdf, png = save_fig(fig, "fig01_system_overview")
plt.close(fig)
print(f"saved {pdf}")
print(f"saved {png}")
