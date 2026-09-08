"""paper/figures/fig02_lift_path.py — Fig.2 (P3).

Design doc: paper/data_inventory.md §Reconciliation (P0).

CANONICAL 7-WAY Macro-F1 = 33.86 % per-fold mean (logit-mean) / 34.03 %
pooled.  This is the production "Equal"/logit-mean convention.  Every
ensemble stage here is recomputed from the SAME OOF artifacts under ONE
(logit-mean) convention so the lift path is apples-to-apples.  As of
2026-05-20 (E1.3, review consensus), the path is collapsed to four
stages: STGCN++ baseline → best single → 7-way → 11-way logit-mean.  The
intermediate 9-way (+MotionBERT+C3D) and 11-way softmax-mean stages were
dropped from the displayed bars: the +1.15 pp logit-vs-softmax ablation is
reported in §4.2 prose (no softmax-mean absolute number per guide §5), and
the external-pretraining contribution is carried by Fig.4 orthogonality.

Panel A: stage-by-stage Macro-F1, bar = 10-fold LPO per-fold mean,
  whisker = ±1 SD (matches the official baseline convention "25.2 % ±
  4.5 %"); sample-level paired bootstrap 95 % CI (1000×, seed 42) is
  computed and written to fig02_lift_path_stats.json for P9 Table 1.
Panel B: per-class F1 progression (pooled OOF): STGCN++ → best single
  (Region-Aware) → 11-way logit-mean; weakest-improvement class bracketed.

Anonymized (no institution / author / first-person / repo handle).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # paper/

import numpy as np  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

import reconcile_f1 as R  # noqa: E402
from _style import (PALETTE_QUAL, ONE_COL, save_fig, seed_all,  # noqa: E402
                     set_size, setup_ieee)
from diema.data.parser import IDX_TO_EMOTION  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

setup_ieee()
seed_all(42)

raw, y, fold_ids = R.load_oof()
EMO = [IDX_TO_EMOTION[i] for i in range(12)]


def _perfold(pred, yy, fids):
    v = [f1_score(yy[fids == k], pred[fids == k], average="macro")
         for k in range(10)]
    return float(np.mean(v)), float(np.std(v, ddof=1))


def _boot(pred, yy, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    N = len(yy)
    s = np.empty(n)
    for b in range(n):
        i = rng.integers(0, N, N)
        s[b] = f1_score(yy[i], pred[i], average="macro")
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def _exp001():
    """STGCN++ baseline — separate per-fold artifacts (own oof_labels)."""
    d = R.ART / "exp001_benchmark_repro"
    P, Y, F = [], [], []
    for fid in range(10):
        L = np.load(d / f"fold_{fid:02d}/oof_logits.npy")
        yy = np.load(d / f"fold_{fid:02d}/oof_labels.npy")
        P.append(L.argmax(-1))
        Y.append(yy)
        F.append(np.full(len(yy), fid))
    return np.concatenate(P), np.concatenate(Y), np.concatenate(F)


b_pred, b_y, b_fid = _exp001()

# ── stages: (label, pred, y, fold_ids) ────────────────────────────────────
# 2026-05-20 (E1.3): collapsed 6 → 4 stages; see file docstring.
STAGES = [
    ("STGCN++\nbaseline", b_pred, b_y, b_fid),
    ("best single\n(Region-Aware)",
     R.combine(raw, ["exp034_regionaware_convtr_a00"], "logit_mean"), y, fold_ids),
    ("7-way\nensemble", R.combine(raw, R.PROD, "logit_mean"), y, fold_ids),
    ("11-way\nlogit-mean",
     R.combine(raw, R.PROD + R.EXTRA4, "logit_mean"), y, fold_ids),
]

means, sds, ci_lo, ci_hi, pooled = [], [], [], [], []
stats = {}
for label, pred, yy, fids in STAGES:
    m, s = _perfold(pred, yy, fids)
    lo, hi = _boot(pred, yy)
    p = float(f1_score(yy, pred, average="macro"))
    means.append(m * 100)
    sds.append(s * 100)
    ci_lo.append(lo * 100)
    ci_hi.append(hi * 100)
    pooled.append(p * 100)
    stats[label.replace("\n", " ")] = {
        "perfold_mean_pct": m * 100, "perfold_sd_pct": s * 100,
        "pooled_pct": p * 100, "boot95_ci_pct": [lo * 100, hi * 100]}

# ── figure: 2 stacked panels, 1-column ────────────────────────────────────
fig, (axA, axB) = plt.subplots(
    2, 1, gridspec_kw={"height_ratios": [1.0, 1.18], "hspace": 0.75})
set_size(fig, ONE_COL, 5.6)
SHORT = ["STGCN++", "Best single", "7-way", "11-way\nlogit-mean"]

# Panel A
x = np.arange(len(STAGES))
axA.bar(x, means, width=0.62, color=PALETTE_QUAL[0],
        edgecolor="0.25", linewidth=0.5,
        yerr=sds, error_kw=dict(elinewidth=0.6, capsize=2, capthick=0.6,
                                ecolor="0.2"))
for xi, m, s in zip(x, means, sds):
    axA.text(xi, m + s + 0.5, f"{m:.1f}±{s:.1f}", ha="center", va="bottom",
             fontsize=6.0)
axA.set_xticks(x)
axA.set_xticklabels(SHORT, rotation=28, ha="right", fontsize=5.8)
axA.set_ylabel("Macro-F1 (%)")
axA.set_ylim(min(means) - 3.0, max(means) + max(sds) + 3.0)
axA.set_title("(A) Stage-by-stage lift (10-fold LPO mean ± SD)",
              fontsize=8, loc="left")
axA.grid(axis="y", alpha=0.3, linewidth=0.3)
axA.grid(axis="x", visible=False)

# Panel B — per-class F1 progression (pooled OOF)
def _pc(pred, yy):
    return f1_score(yy, pred, average=None, labels=list(range(12))) * 100

pc_base = _pc(b_pred, b_y)
pc_single = _pc(R.combine(raw, ["exp034_regionaware_convtr_a00"],
                          "logit_mean"), y)
pc_ens = _pc(R.combine(raw, R.PROD + R.EXTRA4, "logit_mean"), y)

xb = np.arange(12)
w = 0.27
tiers = [("STGCN++ baseline", pc_base, PALETTE_QUAL[0]),
         ("best single (Region-Aware)", pc_single, PALETTE_QUAL[1]),
         ("11-way logit-mean", pc_ens, PALETTE_QUAL[2])]
for j, (lab, vals, col) in enumerate(tiers):
    axB.bar(xb + (j - 1) * w, vals, width=w, color=col, label=lab,
            edgecolor="0.3", linewidth=0.3)
axB.set_xticks(xb)
axB.set_xticklabels(EMO, rotation=45, ha="right", fontsize=6.0)
axB.set_ylabel("per-class F1 (%)")
axB.set_ylim(0, max(pc_ens.max(), pc_single.max(), pc_base.max()) + 12)
axB.set_title("(B) Per-class F1 progression (pooled OOF)", fontsize=8,
              loc="left")
axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3,
           fontsize=5.4, handlelength=1.2, columnspacing=0.9,
           borderaxespad=0.0, frameon=False)
axB.grid(axis="y", alpha=0.3, linewidth=0.3)
axB.grid(axis="x", visible=False)

# weakest-improvement class: min (11-way − baseline)
delta = pc_ens - pc_base
wc = int(np.argmin(delta))
ytop = max(pc_base[wc], pc_single[wc], pc_ens[wc]) + 2.0
axB.annotate(
    f"weakest Δ\n{EMO[wc]} {delta[wc]:+.1f}pp",
    xy=(wc, ytop), xytext=(wc, ytop + 9),
    ha="center", va="bottom", fontsize=5.6, color="0.15",
    arrowprops=dict(arrowstyle="-[,widthB=1.6", lw=0.6, color="0.3"))

pdf, png = save_fig(fig, "fig02_lift_path")
plt.close(fig)
Path(Path(__file__).resolve().parent / "fig02_lift_path_stats.json").write_text(
    json.dumps({"canonical_7way": "33.86% per-fold / 34.03% pooled "
                "(logit-mean convention)",
                "stages": stats,
                "weakest_improvement_class": EMO[wc],
                "weakest_delta_pp": float(delta[wc])}, indent=2))
# 2026-05-20 (post-E1.3, a review): the previous canonical_7way value
# contained the literal '33.72% softmax-mean' as a negation — the only
# machine-readable softmax token left in any artifact. Removed per guide §5
# ("softmax-mean appears NOWHERE, not even a footnote").
print(f"saved {pdf}")
print(f"saved {png}")
print(f"weakest-improvement class: {EMO[wc]} ({delta[wc]:+.2f} pp)")
for k, v in stats.items():
    print(f"  {k:28s} perfold {v['perfold_mean_pct']:.2f}±"
          f"{v['perfold_sd_pct']:.2f}  pooled {v['pooled_pct']:.2f}  "
          f"CI95 [{v['boot95_ci_pct'][0]:.2f},{v['boot95_ci_pct'][1]:.2f}]")

# ── (P2b) compact body variant — Panel A only, landscape ──────────────────
# The full figure above is portrait (ONE_COL x 5.6 in) and is too tall for the
# 7-page camera-ready body.  The body carries this compact single-panel
# variant; the per-class progression (Panel B) stays in the supplement, which
# keeps using fig02_lift_path.{pdf,png}.  Same stage values and same bar
# colour; the internal axes title is dropped because the LaTeX caption carries
# it.  BODY_H is the height knob: shorter = safer for the body page budget.
BODY_H = 1.55  # inch

figA2, axA2 = plt.subplots(1, 1)
set_size(figA2, ONE_COL, BODY_H)
axA2.bar(x, means, width=0.62, color=PALETTE_QUAL[0],
         edgecolor="0.25", linewidth=0.5,
         yerr=sds, error_kw=dict(elinewidth=0.6, capsize=2, capthick=0.6,
                                 ecolor="0.2"))
for xi, m, s in zip(x, means, sds):
    axA2.text(xi, m + s + 0.5, f"{m:.1f}±{s:.1f}", ha="center", va="bottom",
              fontsize=6.5)
SHORT_BODY = ["STGCN++", "Best single\n(Region-Aware)", "7-way", "11-way\nlogit-mean"]
axA2.set_xticks(x)
axA2.set_xticklabels(SHORT_BODY, fontsize=6.5)   # horizontal: rotation costs height
axA2.set_ylabel("Macro-F1 (%)", fontsize=8)
axA2.set_ylim(min(m - s for m, s in zip(means, sds)) - 1.5, max(means) + max(sds) + 3.5)
axA2.tick_params(axis="y", labelsize=7)
axA2.grid(axis="y", alpha=0.3, linewidth=0.3)
axA2.grid(axis="x", visible=False)
pdfA, pngA = save_fig(figA2, "fig02_lift_path_bodyA")
plt.close(figA2)
print(f"saved {pdfA}")
print(f"saved {pngA}")
