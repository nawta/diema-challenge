"""paper/figures/fig05_saliency.py — Fig.5 (P6).

Design doc: paper/data_inventory.md (P0, item j + Open Q3).
ACII Bonus Task: visualise temporal / spatial attention (saliency) maps.
Source: experiments/exp078_temporal_spatial_saliency/results.npz
  (saliency = |d logit_true / d x|), keys temporal_sal (12,64),
  spatial_sal (12,25), joint_time_sal (12,64,25), emotion_labels (12),
  joint_names (25).

OPEN Q3 RESOLVED: the prompt/caption draft said temporal entropy ~98.7 %
of log(64).  Recomputed from the artifact it is **99.54 % (range
99.35-99.82 %)** — even MORE uniform.  We caption the real number.

Panels: (A) temporal heatmap + per-emotion entropy inset (the load-bearing
NEGATIVE: near-uniform), (B) spatial heatmap (row-normalised, joints
grouped into the 6 DIEMA body parts, top-1 joint dotted), (C) the single
least-uniform emotion (surprise) joint×time map showing local structure
survives class-averaged temporal uniformity.  Supplementary: the full
12-emotion joint×time grid -> sup_fig05_full_grid.{pdf,png}.

Colourblind-safe (viridis only; NO red-green).  Anonymized.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # paper/

import numpy as np  # noqa: E402

from _style import (PALETTE_SEQ, TWO_COL, save_fig, seed_all,  # noqa: E402
                     setup_ieee)
from diema.models.skeleton_graph import DIEMA_BODY_PARTS  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

setup_ieee()
seed_all(42)

NPZ = "experiments/exp078_temporal_spatial_saliency/results.npz"
d = np.load(NPZ, allow_pickle=True)
EMO = [str(s) for s in d["emotion_labels"]]
JN = [str(s) for s in d["joint_names"]]
temporal = d["temporal_sal"].astype(float)     # (12, 64) mean over joints
spatial = d["spatial_sal"].astype(float)       # (12, 25) mean over time
jt = d["joint_time_sal"].astype(float)         # (12, 64, 25)
T = temporal.shape[1]

# emotion sort: 6 basic (Ekman) then 6 social/self-conscious
BASIC = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
SOCIAL = ["contempt", "jealousy", "shame", "guilt", "gratitude", "pride"]
EORDER = [EMO.index(e) for e in BASIC + SOCIAL]
ELAB = [EMO[i] for i in EORDER]

# joints reordered & grouped by the 6 DIEMA body parts (+ block bounds)
PARTS = list(DIEMA_BODY_PARTS.keys())
JORDER, BOUNDS, PMID, off = [], [], [], 0
for p in PARTS:
    idx = list(DIEMA_BODY_PARTS[p])
    JORDER += idx
    PMID.append(off + len(idx) / 2.0)
    off += len(idx)
    BOUNDS.append(off)
assert len(JORDER) == 25

# per-emotion temporal entropy (normalised by log T) — the NEGATIVE result
pt = temporal / temporal.sum(1, keepdims=True)
H = -(pt * np.log(pt + 1e-12)).sum(1) / np.log(T)
H_mean = float(H.mean())

# least-uniform emotion = lowest entropy of joint-mean temporal profile
tj = jt.mean(2)
ptj = tj / tj.sum(1, keepdims=True)
Hj = -(ptj * np.log(ptj + 1e-12)).sum(1) / np.log(T)
LEAST = int(Hj.argmin())                         # = surprise

# ── main figure: 2-col, 3 panels ──────────────────────────────────────────
fig = plt.figure()
fig.set_size_inches(TWO_COL, 5.1)
gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.30, 1.18],
                      height_ratios=[1.0, 1.05], wspace=0.5, hspace=0.62)
axA = fig.add_subplot(gs[0, 0])
axE = fig.add_subplot(gs[0, 1])
axB = fig.add_subplot(gs[0, 2])
axC = fig.add_subplot(gs[1, :])


def _yemo(ax):
    ax.set_yticks(range(12))
    ax.set_yticklabels(ELAB, fontsize=5.0)
    ax.axhline(5.5, color="white", lw=0.8)        # basic | social divider


# Panel A: temporal heatmap (rows = emotion, cols = frame)
A = temporal[EORDER]
imA = axA.imshow(A, cmap=PALETTE_SEQ, aspect="auto",
                 extent=(0, T, 11.5, -0.5))
axA.set_xlabel("frame (0–64)", fontsize=7)
axA.set_title("(A) Temporal saliency — near-uniform", fontsize=8,
              loc="left")
_yemo(axA)
fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.03).ax.tick_params(
    labelsize=4.5)

# entropy inset (uniform => bars hug 1.0)
axE.barh(range(12), H[EORDER], height=0.7, color="0.45")
axE.axvline(1.0, color="0.15", lw=0.7, ls="--")
axE.set_xlim(0.985, 1.001)
axE.set_ylim(11.5, -0.5)
axE.set_yticks([])
axE.set_xticks([0.99, 1.0])
axE.tick_params(labelsize=4.5)
axE.set_title("H / log T", fontsize=6)
axE.grid(False)

# Panel B: spatial heatmap, joints grouped by part, row-normalised
B = spatial[EORDER][:, JORDER]
B = B / B.max(1, keepdims=True)
imB = axB.imshow(B, cmap=PALETTE_SEQ, aspect="auto",
                 extent=(0, 25, 11.5, -0.5))
for b in BOUNDS[:-1]:
    axB.axvline(b, color="white", lw=0.7)
for m, p in zip(PMID, PARTS):                    # part labels BELOW heatmap
    axB.text(m, 12.35, p, ha="center", va="top", fontsize=4.8,
             color="0.3", rotation=45, clip_on=False)
# top-1 joint per emotion (dot)
for r in range(12):
    c = int(np.argmax(B[r]))
    axB.plot(c + 0.5, r, "o", ms=2.0, mfc="white", mec="black", mew=0.4)
axB.set_xticks([])
axB.set_title("(B) Spatial saliency (row-norm.; •=top-1 joint)",
              fontsize=8, loc="left")
_yemo(axB)
fig.colorbar(imB, ax=axB, fraction=0.046, pad=0.03).ax.tick_params(
    labelsize=4.5)

# Panel C: single least-uniform emotion, joint×time
C = jt[LEAST][:, JORDER].T                       # (25 joints, 64 frames)
imC = axC.imshow(C, cmap=PALETTE_SEQ, aspect="auto",
                 extent=(0, T, 25, 0))
for b in BOUNDS[:-1]:
    axC.axhline(b, color="white", lw=0.7)
for m, p in zip(PMID, PARTS):
    axC.text(-1.8, m, p, ha="right", va="center", fontsize=5.4,
             color="0.3", clip_on=False)
axC.set_xlabel("frame (0–64)", fontsize=7)
axC.set_yticks([])
axC.set_title(f"(C) Joint×frame saliency — {EMO[LEAST]} "
              f"(least-uniform, H/logT={Hj[LEAST]:.3f}): local structure "
              f"survives class-averaged uniformity", fontsize=7, loc="left")
fig.colorbar(imC, ax=axC, fraction=0.020, pad=0.015).ax.tick_params(
    labelsize=4.5)

pdf, png = save_fig(fig, "fig05_saliency")
plt.close(fig)

# ── supplementary: full 12-emotion joint×time grid ───────────────────────
figS, axS = plt.subplots(3, 4)
figS.set_size_inches(TWO_COL, 6.4)
vmax = float(jt.max())
for k, ei in enumerate(EORDER):
    ax = axS[k // 4][k % 4]
    ax.imshow(jt[ei][:, JORDER].T, cmap=PALETTE_SEQ, aspect="auto",
              vmin=0, vmax=vmax, extent=(0, T, 25, 0))
    for b in BOUNDS[:-1]:
        ax.axhline(b, color="white", lw=0.4)
    ax.set_title(EMO[ei], fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])
figS.suptitle("Supplementary Fig. 5: joint×frame saliency, all 12 "
              "emotions (joints grouped by body part)", fontsize=8)
figS.subplots_adjust(wspace=0.08, hspace=0.22, top=0.93)
pdfS, pngS = save_fig(figS, "sup_fig05_full_grid")
plt.close(figS)

print(f"saved {pdf}")
print(f"saved {png}")
print(f"saved {pdfS}")
print(f"saved {pngS}")
print(f"temporal entropy / log(64): mean {H_mean:.4f} "
      f"(range {H.min():.4f}-{H.max():.4f})  [prompt draft said 0.987 — "
      f"ACTUAL is {H_mean:.3f}, MORE uniform]")
print(f"least-uniform emotion = {EMO[LEAST]} (H/logT={Hj[LEAST]:.4f})")
