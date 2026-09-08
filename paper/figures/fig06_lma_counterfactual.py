"""paper/figures/fig06_lma_counterfactual.py — Fig.6 (P7).

Design doc: paper/data_inventory.md (P0, items i + d).  Best-Explainability
load-bearing figure: triangulates "what humans call body language" (LMA,
Panel A) with "what happens when we perturb the motion" (counterfactual
edits, Panel B), anchored by the region saliency-LMA Spearman.

Panel A: 12 emotion × 32 LMA z-score heatmap (RdBu_r, centred 0), attrs
  grouped Body/Effort/Shape/Space; right strip = per-emotion saliency-LMA
  Spearman; inset box contrasts mean LMA rho = +0.500 vs classical
  kinematics +0.033 (exp087 faithfulness_results.json).
Panel B: 6 body-part × 3 edit-type mean delta-p(true class) (RdBu_r,
  centred 0; more negative = more disruptive). Top-3 most-disruptive cells
  annotated; l_arm+amplify flip-rate annotated.  Inset: Spearman between
  per-part deep saliency (exp078) and per-part counterfactual disruption.

CAUSAL-LANGUAGE DISCIPLINE: counterfactual freeze/damp/amplify are
observational perturbations ("edits", "interventions", "disrupts"), NOT
randomised causal experiments — no "causes" anywhere.

Anonymized.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # paper/

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from _style import (PALETTE_DIV, PALETTE_SEQ, TWO_COL, save_fig,  # noqa: E402
                     seed_all, setup_ieee)
from diema.data.parser import IDX_TO_EMOTION  # noqa: E402
from diema.models.skeleton_graph import DIEMA_BODY_PARTS  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

setup_ieee()
seed_all(42)

EMO = [IDX_TO_EMOTION[i] for i in range(12)]

# ── Panel A: LMA z-scores ─────────────────────────────────────────────────
lz = np.load("output/lma_attrs_rule.npz", allow_pickle=True)
feat = lz["features"].astype(float)               # (7992, 32)
lab = lz["labels"].astype(int)
names = [str(s) for s in lz["names"]]
z = (feat - feat.mean(0)) / (feat.std(0) + 1e-9)
Z = np.stack([z[lab == c].mean(0) for c in range(12)])   # (12, 32)
GROUPS = [("Body", 0, 8), ("Effort", 8, 16), ("Shape", 16, 24),
          ("Space", 24, 32)]

fj = json.load(open("experiments/exp087_lma_attr_aux/"
                     "faithfulness_results.json"))
RHO_LMA = fj["lma_region_faithfulness_spearman_mean"]      # 0.500
RHO_CLS = fj["exp079_baseline_spearman_mean"]              # 0.033
per_lma = np.array([fj["per_emotion_rho"][e]["lma_rho"] for e in EMO])

# ── Panel B: counterfactual edits ─────────────────────────────────────────
PARTS = ["head", "torso", "l_arm", "r_arm", "l_leg", "r_leg"]
EDITS = ["freeze", "damp_0.5", "amplify_2.0"]
dlt = defaultdict(list)
flp = defaultdict(list)
with open("docs/analysis/counterfactual/"
          "exp034_regionaware_convtr_a00_fold00_counterfactual.csv") as f:
    for r in csv.DictReader(f):
        dlt[(r["part"], r["edit"])].append(float(r["delta_p_true"]))
        flp[(r["part"], r["edit"])].append(r["flipped"] == "True")
CF = np.array([[np.mean(dlt[(p, e)]) for e in EDITS] for p in PARTS])
FLIP = np.array([[100 * np.mean(flp[(p, e)]) for e in EDITS] for p in PARTS])
# top-3 most disruptive (most negative) cells
flat = np.argsort(CF, axis=None)[:3]
top3 = [(int(k // 3), int(k % 3)) for k in flat]

# triangulation: per-part deep saliency (exp078) vs per-part disruption
sp = np.load("experiments/exp078_temporal_spatial_saliency/results.npz",
             allow_pickle=True)["spatial_sal"].astype(float).mean(0)  # (25,)
sal_part = np.array([sp[list(DIEMA_BODY_PARTS[p])].mean() for p in PARTS])
disr_part = -CF.mean(1)                                     # higher = worse
tri_rho, tri_p = spearmanr(sal_part, disr_part)

# ── figure ────────────────────────────────────────────────────────────────
fig = plt.figure()
fig.set_size_inches(TWO_COL, 3.8)
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.12, 0.52], wspace=0.55)
axA = fig.add_subplot(gs[0, 0])
axR = fig.add_subplot(gs[0, 1])
axB = fig.add_subplot(gs[0, 2])

mz = float(np.abs(Z).max())
imA = axA.imshow(Z, cmap=PALETTE_DIV, vmin=-mz, vmax=mz, aspect="auto")
for _, a, b in GROUPS[:-1]:
    axA.axvline(b - 0.5, color="0.2", lw=0.7)
for nm, a, b in GROUPS:                          # group labels BELOW heatmap
    axA.text((a + b - 1) / 2, 12.05, nm, ha="center", va="top",
             fontsize=5.8, color="0.3", clip_on=False)
axA.set_xticks([])
axA.set_yticks(range(12))
axA.set_yticklabels(EMO, fontsize=5.4)
axA.set_title("(A) LMA z-score per emotion (Body·Effort·Shape·Space)",
              fontsize=7.5, loc="left")
axA.grid(False)
cbA = fig.colorbar(imA, ax=axA, orientation="horizontal", fraction=0.045,
                   pad=0.13)
cbA.ax.tick_params(labelsize=4.5)
cbA.set_label("mean z", fontsize=5.5)

# per-emotion LMA-saliency rho strip (aligned to rows)
axR.imshow(per_lma[:, None], cmap=PALETTE_SEQ, vmin=0, vmax=1,
           aspect="auto")
for i, v in enumerate(per_lma):
    axR.text(0, i, f"{v:.2f}", ha="center", va="center", fontsize=4.6,
             color="white" if v < 0.6 else "black")
axR.set_xticks([0])
axR.set_xticklabels(["ρ(LMA)"], fontsize=5.0, rotation=90)
axR.set_yticks([])
axR.set_title("saliency–LMA", fontsize=5.6)
axR.grid(False)

# Panel B counterfactual
mb = float(np.abs(CF).max())
imB = axB.imshow(CF, cmap=PALETTE_DIV, vmin=-mb, vmax=mb, aspect="auto")
for i in range(6):
    for j in range(3):
        axB.text(j, i, f"{CF[i, j]:+.4f}", ha="center", va="center",
                 fontsize=4.6,
                 color="white" if abs(CF[i, j]) > mb * 0.55 else "black")
for i, j in top3:
    axB.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                            edgecolor="black", linewidth=1.1))
# l_arm + amplify_2.0 flip-rate annotation (per spec)
la = (PARTS.index("l_arm"), EDITS.index("amplify_2.0"))
axB.annotate(f"flip {FLIP[la]:.1f}%", xy=(la[1], la[0]),
             xytext=(la[1] + 0.05, la[0] - 1.05), fontsize=4.8,
             ha="center", color="0.15",
             arrowprops=dict(arrowstyle="->", lw=0.5, color="0.3"))
axB.set_xticks(range(3))
axB.set_xticklabels(EDITS, rotation=30, ha="right", fontsize=5.6)
axB.set_yticks(range(6))
axB.set_yticklabels(PARTS, fontsize=5.6)
axB.set_title("(B) Counterfactual edits — Δp(true)", fontsize=8,
              loc="left")
axB.grid(False)
cbB = fig.colorbar(imB, ax=axB, fraction=0.05, pad=0.04)
cbB.ax.tick_params(labelsize=4.5)
cbB.set_label("Δp(true)", fontsize=5.5)

# load-bearing contrast + triangulation, as a text box under Panel B
fig.text(0.595, -0.06,
         f"region saliency–LMA Spearman ρ = +{RHO_LMA:.3f}\n"
         f"(classical kinematics: +{RHO_CLS:.3f}, "
         f"{RHO_LMA / RHO_CLS:.0f}× weaker)\n"
         f"triangulation: per-part saliency vs counterfactual\n"
         f"disruption Spearman ρ = {tri_rho:+.2f} (p={tri_p:.2f}, n=6)",
         ha="center", va="top", fontsize=5.6,
         bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4ef", ec="0.6",
                   lw=0.5))

pdf, png = save_fig(fig, "fig06_lma_counterfactual")
plt.close(fig)
print(f"saved {pdf}")
print(f"saved {png}")
print(f"LMA rho={RHO_LMA:.4f}  classical={RHO_CLS:.4f}  "
      f"({RHO_LMA / RHO_CLS:.1f}x)")
print("top-3 most-disruptive (part,edit,Δp):")
for i, j in top3:
    print(f"  {PARTS[i]:>6} + {EDITS[j]:<11} {CF[i, j]:+.4f} "
          f"(flip {FLIP[i, j]:.1f}%)")
print(f"l_arm+amplify flip = {FLIP[la]:.1f}%")
print(f"triangulation saliency~disruption Spearman = {tri_rho:+.3f} "
      f"(p={tri_p:.3f}, n=6 parts)")
