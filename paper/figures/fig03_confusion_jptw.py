"""paper/figures/fig03_confusion_jptw.py — Fig.3 (P4).

Design doc: paper/data_inventory.md (P0).  2-column, two panels.

Panel A: 12x12 OOF confusion of the 11-way logit-mean ensemble, rows = true
  emotion, row-normalized to 100 %, viridis, top-5 off-diagonal boxed.
Panel B: 11-way per-class F1 by performer country (JP vs TW) + signed
  |F1_JP - F1_TW| strip.

Both panels use the SAME 11-way logit-mean ensemble (the paper's headline
model) recomputed from OOF, so A and B are self-consistent.  The exp034
cross-cultural part-masking analysis (docs/.../cross_cultural_README.md) is
a *different* metric (part-masking, single model); its qualitative findings
(shared head->legs->arms->torso ordering; JP-TW macro gap partly a marker-
set confound — JP_01-05 used OptiTrack, not Vicon) are cited in the caption,
not plotted here.

Anonymized: country aggregates only, no individual performer IDs.
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # paper/

import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

import reconcile_f1 as R  # noqa: E402
from _style import (ONE_COL, PALETTE_QUAL, PALETTE_SEQ, PALETTE_DIV,  # noqa
                     TWO_COL, save_fig, seed_all, setup_ieee)
from diema.data.parser import IDX_TO_EMOTION  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

setup_ieee()
seed_all(42)

raw, y, fold_ids = R.load_oof()
EMO = [IDX_TO_EMOTION[i] for i in range(12)]

# country aligned to y — re-derive ref EXACTLY as reconcile_f1.load_oof()
with open(R.SPLITS_PKL, "rb") as f:
    splits = pickle.load(f)
ref = [str(fn) for fid in range(10) for fn, _ in splits[fid]["val"]]
country = np.array([s.split("_")[0] for s in ref])
assert len(country) == len(y)

pred = R.combine(raw, R.PROD + R.EXTRA4, "logit_mean")  # 11-way logit-mean

# ── Panel A data: row-normalized confusion ────────────────────────────────
cm = confusion_matrix(y, pred, labels=list(range(12))).astype(float)
pct = cm / cm.sum(axis=1, keepdims=True) * 100.0
assert np.allclose(pct.sum(axis=1), 100.0), "rows must sum to 100%"

# top-5 off-diagonal confusion cells (true i -> pred j, i != j)
off = pct.copy()
np.fill_diagonal(off, -1.0)
flat = np.argsort(off, axis=None)[::-1][:5]
top5 = [(int(k // 12), int(k % 12), off.flat[k]) for k in flat]

# ── Panel B data: 11-way per-class F1 by country ──────────────────────────
mj, mt = country == "JP", country == "TW"
f1_jp = f1_score(y[mj], pred[mj], average=None, labels=list(range(12))) * 100
f1_tw = f1_score(y[mt], pred[mt], average=None, labels=list(range(12))) * 100
mac_jp = f1_score(y[mj], pred[mj], average="macro") * 100
mac_tw = f1_score(y[mt], pred[mt], average="macro") * 100
diff = f1_jp - f1_tw                       # signed JP - TW
top3 = np.argsort(np.abs(diff))[::-1][:3]


def _draw_panelA(fig, ax, standalone=False):
    """Draw Panel A (row-normalized OOF confusion) onto ``ax``.

    Factored out of the main 2x2 figure so a standalone ONE_COL asset can
    reuse the identical drawing (fig05_saliency.py has the two-figure
    precedent).  ``fig`` is threaded through only because the colorbar is
    created via ``fig.colorbar(..., ax=ax)`` (a fig-level side effect that
    steals space from ``ax``); passing the caller's fig/ax reproduces the
    original bytes exactly.  ``standalone=True`` drops the axes title (the
    LaTeX caption carries it).  Reuses module-level pct/top5/EMO.
    """
    im = ax.imshow(pct, cmap=PALETTE_SEQ, vmin=0, vmax=pct.max(),
                   aspect="equal")
    for i in range(12):
        for j in range(12):
            v = pct[i, j]
            if v < 0.5:                         # hide ~0 cells
                continue
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.8,
                    color="white" if v < pct.max() * 0.55 else "black")
    for i, j, _v in top5:                        # white box on top-5 confusions
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="white", linewidth=1.1))
    ax.set_xticks(range(12))
    ax.set_xticklabels(EMO, rotation=90, fontsize=7.1)
    ax.set_yticks(range(12))
    ax.set_yticklabels(EMO, fontsize=7.1)
    ax.set_xlabel("predicted", fontsize=8.5)
    ax.set_ylabel("true", fontsize=8.5)
    if not standalone:
        ax.set_title("(A) 11-way OOF confusion (row-normalized %)",
                     fontsize=8, loc="left")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7.1)
    cb.set_label("% of true class", fontsize=7.5)


# ── layout: left = confusion (spans), right = bars over strip ─────────────
fig = plt.figure()
fig.set_size_inches(TWO_COL, 3.7)
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0],
                      height_ratios=[4.0, 1.0], wspace=0.46, hspace=0.16)
axCM = fig.add_subplot(gs[:, 0])
axBar = fig.add_subplot(gs[0, 1])
axStrip = fig.add_subplot(gs[1, 1])

# Panel A
_draw_panelA(fig, axCM, standalone=False)

# Panel B — paired bars
xb = np.arange(12)
w = 0.40
axBar.bar(xb - w / 2, f1_jp, w, color=PALETTE_QUAL[3],
          label=f"JP (n=40 perf.; macro {mac_jp:.1f}%)",
          edgecolor="0.3", linewidth=0.3)
axBar.bar(xb + w / 2, f1_tw, w, color=PALETTE_QUAL[0],
          label=f"TW (n=34 perf.; macro {mac_tw:.1f}%)",
          edgecolor="0.3", linewidth=0.3)
axBar.set_xlim(-0.6, 11.6)
axBar.set_ylim(0, max(f1_jp.max(), f1_tw.max()) * 1.22)  # legend headroom
axBar.set_xticks(xb)
axBar.set_xticklabels([])
axBar.set_ylabel("per-class F1 (%)", fontsize=7)
axBar.set_title("(B) Per-class F1 by performer country", fontsize=8,
                loc="left")
axBar.legend(loc="upper left", fontsize=4.8, handlelength=1.1,
             labelspacing=0.3, borderpad=0.4, framealpha=0.85,
             edgecolor="0.7")
axBar.grid(axis="y", alpha=0.3, linewidth=0.3)
axBar.grid(axis="x", visible=False)

# Panel B — signed |JP-TW| strip
M = float(np.abs(diff).max())
axStrip.imshow(diff[None, :], cmap=PALETTE_DIV, vmin=-M, vmax=M,
               aspect="auto", extent=(-0.5, 11.5, -0.5, 0.5))
for k in top3:
    axStrip.text(k, 0, f"{diff[k]:+.0f}", ha="center", va="center",
                 fontsize=5.0, fontweight="bold",
                 color="white" if abs(diff[k]) > M * 0.55 else "black")
axStrip.set_yticks([])
axStrip.set_xticks(xb)
axStrip.set_xticklabels(EMO, rotation=45, ha="right", fontsize=5.4)
axStrip.set_xlim(-0.6, 11.6)
axStrip.set_ylabel("JP−TW\n(pp)", fontsize=5.6, rotation=0, ha="right",
                   va="center")
axStrip.grid(False)

pdf, png = save_fig(fig, "fig03_confusion_jptw")
plt.close(fig)

# ── standalone Panel-A-only asset (ONE_COL) ───────────────────────────────
# Native panel-A figure so the camera-ready can drop the LaTeX \trim hack
# that currently crops Fig.3 down to just the confusion matrix.
figA = plt.figure()
figA.set_size_inches(ONE_COL, 3.4)
axA = figA.add_subplot(111)
_draw_panelA(figA, axA, standalone=True)
pdfA, pngA = save_fig(figA, "fig03_confusion_jptw_panelA")
plt.close(figA)

print(f"saved {pdf}")
print(f"saved {png}")
print(f"saved {pdfA}")
print(f"saved {pngA}")
print(f"JP macro {mac_jp:.2f}%  TW macro {mac_tw:.2f}%  "
      f"(11-way; exp034-part-masking ref: JP 28.58 / TW 31.28)")
print("top-5 off-diagonal confusions (true -> pred, %):")
for i, j, v in top5:
    print(f"  {EMO[i]:>9} -> {EMO[j]:<9} {v:5.1f}%")
print("top-3 |JP-TW| per-class (pp): "
      + ", ".join(f"{EMO[k]} {diff[k]:+.1f}" for k in top3))
