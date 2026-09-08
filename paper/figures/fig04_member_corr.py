"""paper/figures/fig04_member_corr.py — Fig.4 (P5).

Design doc: paper/data_inventory.md (P0).  The load-bearing figure for
"why error-space orthogonality matters": pairwise correlation of per-sample
ERROR indicators across the 11 ensemble members (OOF, n=7,992).

For binary indicators the Pearson correlation IS the phi coefficient; we
compute both and assert equality, then plot/emit the (identical) matrix.

Members ordered semantically into 3 groups (thick black block borders):
  G1 skeleton-only base : STGCN++, CTR-GCN, SkateFormer, ProtoGCN
  G2 skeleton hybrid    : Conv1D+Tr, Region-Aware, KP-MLP
  G3 external pretrain  : MotionBERT, C3D-stats, MAMP-NTU60, MAMP-NTU120

All 11 member OOF files present (P0 inventory) -> full 11x11, no skips.
Anonymized (model family names only).
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # paper/

import numpy as np  # noqa: E402

import reconcile_f1 as R  # noqa: E402
from _style import (PALETTE_SEQ, ONE_COL, save_fig, seed_all,  # noqa: E402
                     setup_ieee)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

setup_ieee()
seed_all(42)

raw, y, fold_ids = R.load_oof()

# semantic member order + display labels + group spans
ORDER = ["exp002_a04_smooth", "exp003_ctrgcn_a00", "exp004_skateformer_a01",
         "exp006_protogcn_a00", "exp020_conv1d_transformer_a01",
         "exp034_regionaware_convtr_a00", "exp023_keypoint_pool_mlp_a00",
         "MB", "C3D", "MAMP60xsub", "MAMP120xset"]
LABELS = ["STGCN++", "CTR-GCN", "SkateFormer", "ProtoGCN", "Conv1D+Tr",
          "Region-Aware", "KP-MLP", "MotionBERT", "C3D-stats", "MAMP-NTU60",
          "MAMP-NTU120"]
GROUPS = [(0, 4, "skeleton base"), (4, 7, "skeleton hybrid"),
          (7, 11, "external pretrain")]
assert set(ORDER) == set(R.PROD + R.EXTRA4) and len(ORDER) == 11

# per-sample error indicator e_m[i] = 1 if member m wrong on sample i
E = np.stack([(raw[m].argmax(-1) != y).astype(np.float64) for m in ORDER])
pearson = np.corrcoef(E)                       # 11x11, diag = 1


def _phi(a, b):
    """phi coefficient from the 2x2 binary contingency table."""
    n11 = np.sum((a == 1) & (b == 1))
    n10 = np.sum((a == 1) & (b == 0))
    n01 = np.sum((a == 0) & (b == 1))
    n00 = np.sum((a == 0) & (b == 0))
    den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return (n11 * n00 - n10 * n01) / den if den > 0 else 0.0


phi = np.array([[_phi(E[i], E[j]) for j in range(11)] for i in range(11)])
assert np.allclose(pearson, phi, atol=1e-9), "phi must equal Pearson (binary)"

# reverse-rank off-diagonal pairs (upper triangle)
pairs = [(i, j, pearson[i, j]) for i in range(11) for j in range(i + 1, 11)]
pairs.sort(key=lambda t: t[2])
low3 = pairs[:3]
high3 = pairs[-3:][::-1]

# machine-readable CSV
csv_path = Path(__file__).resolve().parent / "fig04_member_corr_values.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["# Pearson==phi (per-sample error indicators, OOF n=%d)"
                % len(y)])
    w.writerow([""] + LABELS)
    for i, lab in enumerate(LABELS):
        w.writerow([lab] + [f"{pearson[i, j]:.4f}" for j in range(11)])

# off-diagonal max -> vmax (don't waste cmap on the trivial diagonal=1)
offmax = float(np.max(pearson - np.eye(11) * 2))

fig, ax = plt.subplots()
fig.set_size_inches(ONE_COL, 3.7)
im = ax.imshow(pearson, cmap=PALETTE_SEQ, vmin=0.0, vmax=offmax,
               aspect="equal")
for i in range(11):
    for j in range(11):
        v = pearson[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.6,
                color="white" if v < offmax * 0.6 else "black")
# thick group block borders
for a, b, _ in GROUPS:
    ax.add_patch(Rectangle((a - 0.5, a - 0.5), b - a, b - a, fill=False,
                           edgecolor="black", linewidth=1.0))
ax.set_xticks(range(11))
ax.set_xticklabels(LABELS, rotation=90, fontsize=5.2)
ax.set_yticks(range(11))
ax.set_yticklabels(LABELS, fontsize=5.2)
ax.grid(False)
# group brackets along the top
line_y = -0.75
for a, b, _ in GROUPS:
    ax.plot([a - 0.45, b - 0.55], [line_y, line_y], lw=1.0, color="0.3",
            clip_on=False)
offset = 0.22
texts = []
for a, b, name in GROUPS:
    texts.append(ax.text((a + b - 1) / 2, line_y - offset, name,
                         ha="center", va="bottom", fontsize=5.0,
                         color="0.3", clip_on=False))
for _ in range(5):
    fig.canvas.draw()
    line_y_display = ax.transData.transform((0, line_y))[1]
    low = line_y_display - 3.0
    high = line_y_display + 3.0
    overlaps = [
        text.get_window_extent(renderer=fig.canvas.get_renderer()).y0 <= high
        and text.get_window_extent(renderer=fig.canvas.get_renderer()).y1 >= low
        for text in texts
    ]
    if not any(overlaps):
        break
    offset += 0.12
    for text in texts:
        text.set_y(line_y - offset)
else:
    raise AssertionError("Group labels overlap bracket line after 5 offset attempts")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.ax.tick_params(labelsize=5)
cb.set_label("error-correlation ρ (Pearson = phi)", fontsize=6)

pdf, png = save_fig(fig, "fig04_member_corr")
plt.close(fig)
print(f"saved {pdf}")
print(f"saved {png}")
print(f"saved {csv_path}")
print(f"off-diagonal range: [{min(p[2] for p in pairs):.3f}, "
      f"{max(p[2] for p in pairs):.3f}]")
print("3 MOST orthogonal (lowest ρ):")
for i, j, v in low3:
    print(f"  {LABELS[i]:>12} ~ {LABELS[j]:<12} ρ={v:.3f}")
print("3 MOST redundant (highest ρ):")
for i, j, v in high3:
    print(f"  {LABELS[i]:>12} ~ {LABELS[j]:<12} ρ={v:.3f}")
