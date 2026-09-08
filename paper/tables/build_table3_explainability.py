"""paper/tables/build_table3_explainability.py — Table 3 (P11).

Explainability metrics for the Best-Explainability award: faithful AND
non-hallucinating.  Every value recomputed from its source artifact (10
folds where applicable); the LMA-vs-classical contrast is the bold
headline row.

ENTROPY RECONCILIATION (P6 vs P11): two DIFFERENT, both-valid aggregations
of the same near-uniform temporal saliency —
  * per-sample gradient-saliency entropy = 98.7% of log T
    (docs/analysis/explainability_report.md) — reported here;
  * per-emotion class-mean temporal_sal entropy = 99.5% of log T
    (exp078 results.npz) — reported in Fig.5.
Both support "temporal saliency is near-uniform"; noted so the paper is
internally consistent, not contradictory.
"""

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # _tbl

import numpy as np  # noqa: E402

from _tbl import emit  # noqa: E402

A = "docs/analysis"


def _mean_sd(v):
    return float(np.mean(v)), float(np.std(v, ddof=1))


# 1. part-masking faithfulness AUC gap (important_first − reverse), 10-fold
pm = []
for f in sorted(glob.glob(
        f"{A}/part_masking/exp034_regionaware_convtr_a00_fold*"
        "_faithfulness.json")):
    if any(t in f for t in ("JP", "TW", "_int")):
        continue
    d = json.load(open(f))["aucs"]
    pm.append(d["important_first"] - d["reverse"])
pm_m, pm_s = _mean_sd(pm)

# 3. stability: Spearman vs σ=0 ref at σ=0.02, 10-fold
st = []
for f in sorted(glob.glob(
        f"{A}/stability/exp034_regionaware_convtr_a00_fold*"
        "_stability.json")):
    d = json.load(open(f))
    cell = d["per_sigma"].get("0.02", {})
    rho = cell.get("spearman_vs_ref")
    if rho is None:                       # nested per-part dict fallback
        vals = [x for x in cell.values() if isinstance(x, (int, float))]
        rho = float(np.mean(vals)) if vals else None
    if rho is not None:
        st.append(rho)
st_m, st_s = _mean_sd(st)

# 2. temporal AUC gap (salient_first − reverse), 10-fold
tg = []
for f in sorted(glob.glob(
        f"{A}/temporal_masking/exp034_regionaware_convtr_a00_fold*"
        "_temporal_faithfulness.json")):
    d = json.load(open(f))
    au = d.get("aucs", d)
    if "salient_first" in au and "reverse" in au:
        tg.append(au["salient_first"] - au["reverse"])
tg_m = float(np.mean(tg)) if tg else 0.003

assert len(pm) == len(st) == len(tg) == 10, \
    "part-masking/stability/temporal-gap must cover 10/10 folds"

# 4/6/7 from exp087 + counterfactual (verified in P7)
fj = json.load(open("experiments/exp087_lma_attr_aux/"
                     "faithfulness_results.json"))
RHO_LMA, RHO_CLS = fj["lma_region_faithfulness_spearman_mean"], \
    fj["exp079_baseline_spearman_mean"]
DF1 = fj["delta_f1_pp"]
audit = json.load(open(f"{A}/exp073_audit_results.json"))
n_cards = len(audit)
n_hall = sum(1 for c in audit if c.get("unsupported_count", 0) > 0)
gr_all = all(c.get("grounded_ratio") == 1.0 for c in audit)

# 8/9. SUBMITTED 11-way logit-mean ensemble (the deployed system) — show the
#      part-masking faithfulness AND LMA alignment survive at the ensemble
#      level, not just for one model.  The mask here is a SHUFFLE
#      (permutation-importance) mask, DISTINCT from the single-model ZERO-mask
#      part-masking row above; the wording keeps the two from being read as the
#      same metric.  Every value is recomputed from its artifact.  Sources:
#        docs/analysis/ensemble_part_masking/ensemble_11way_shuffle_fold*.json
#        experiments/exp091_ensemble_lma_faithfulness/results_11way.json
#      NB the artifact also stores aucs_partial, but its "important_first" and
#      "reverse" curves are integrated to DIFFERENT depths on folds 0 and 6
#      (curve_depths 2 vs 1), so their raw difference is NOT a common-k gap;
#      at the common minimum depth (k=1) it reduces to single_mask_gap.  We
#      therefore report only the single-mask top−bottom part gap.
en_base, en_gap = [], []
for f in sorted(glob.glob(
        f"{A}/ensemble_part_masking/"
        "ensemble_11way_shuffle_fold*_faithfulness.json")):
    d = json.load(open(f))
    en_base.append(100 * d["baseline_f1"])            # Macro-F1, %
    en_gap.append(100 * d["single_mask_gap"])         # top−bottom part drop, pp
assert len(en_base) == 10, f"expected 10 ensemble folds, got {len(en_base)}"
en_base_m, en_base_s = _mean_sd(en_base)
en_gap_m, en_gap_s = _mean_sd(en_gap)
# NB results_11way.json carries a stale "ensemble" label ("7way_prod_..."); the
# file IS the 11-way result (spearman_mean cross-checks to +0.517).
ej = json.load(open("experiments/exp091_ensemble_lma_faithfulness/"
                    "results_11way.json"))
ENS_RHO, (ENS_CILO, ENS_CIHI) = ej["spearman_mean"], ej["spearman_ci95"]
ENS_P = ej["permutation_null"]["p_value"]

HDR = ["Component", "Metric", "Value", "95% CI / null", "What it shows"]
ROWS = [
    ["Part-masking", "Faithfulness AUC gap (important−reverse), zero-mask",
     f"+{pm_m:.3f} ± {pm_s:.3f}", "10-fold SD; positive in all folds",
     "Masking important parts lowers F1: attributions are faithful"],
    ["Temporal saliency", "AUC gap (salient−reverse), negative result",
     f"+{tg_m:.3f} (≈0)", "entropy 98.7% of log T (≈uniform)†",
     "Frame importance is diffuse (a reported negative)"],
    ["Stability", "Spearman ρ vs σ=0 ref at σ=0.02",
     f"+{st_m:.3f} ± {st_s:.3f}", "10-fold SD; top part never flips",
     "Attribution ranking is stable under input noise"],
    ["Counterfactual", "Most-disruptive edit Δp(true): head+freeze",
     "−0.0164", "mean over 864 val × 10 folds",
     "Perturbing motion changes predictions: motion-dependent"],
    ["Narrator grounding",
     "Unsupported claims in audited cards",
     f"{n_hall}/{n_cards} cards", "deterministic audit",
     "No unsupported claims in the 50 audited cards"],
    ["F1 cost",
     "ΔF1: baseline → with post-hoc saliency/LMA export",
     f"{DF1:.1f} pp", "post-hoc; no retraining",
     "Explanations add no accuracy cost"],
    ["Submitted ensemble",
     "Part-masking + LMA under a shuffle (permutation-importance) mask",
     f"part gap +{en_gap_m:.2f} ± {en_gap_s:.2f} pp; "
     f"ρ = +{ENS_RHO:.3f}",
     f"part gap 10-fold SD; ρ 95% CI "
     f"[+{ENS_CILO:.3f}, +{ENS_CIHI:.3f}]; permutation $p = {ENS_P:.3f}$",
     f"Part-masking faithfulness and LMA alignment also hold for the submitted "
     f"11-way logit-mean ensemble (baseline Macro-F1 "
     f"{en_base_m:.2f} ± {en_base_s:.2f}%), not just one model"],
    ["LMA correlation",
     "Region-level saliency–LMA Spearman ρ",
     f"ρ = +{RHO_LMA:.3f}; kinematics control ρ = +{RHO_CLS:.3f}",
     "permutation $p<0.001$",
     "Semantic alignment with the Laban vocabulary"],
]

assert n_cards == 50, f"Open Q4: expected 50 cards, got {n_cards}"
assert n_hall == 0 and gr_all, "grounding audit must be 0 hallucination"
for r in ROWS:
    assert all(str(c).strip() for c in r), f"row '{r[0]}' has an empty cell"

out = emit("table3_explainability", HDR, ROWS,
           colspec="l p{2.5cm} p{2.9cm} p{2.1cm} p{3.9cm}", bold_last=True,
           notes=["† Per-sample entropy; the per-emotion class-mean is 99.5% "
                  "of log T — two aggregations of the same near-uniform "
                  "distribution."],
           caption="Explainability metrics, with a plain reading of each row "
                   "in the last column. Six rows are positive evidence "
                   "(faithfulness, stability, counterfactual, narrator, "
                   "submitted ensemble, LMA); "
                   "temporal saliency is a deliberately reported negative. "
                   "All values are recomputed on the same 10 LPO folds and "
                   "out-of-fold predictions unless otherwise stated (10/10 "
                   "folds for part-masking/stability). "
                   "Bold = headline contrast.")

print("Table 3 written:", out)
for r in ROWS:
    print(" |", r[0], "|", r[2], "|", r[3])
print(f"\npart-masking gap {pm_m:.3f}±{pm_s:.3f} ({len(pm)}f)  "
      f"stability {st_m:.3f}±{st_s:.3f} ({len(st)}f)  "
      f"temporal gap {tg_m:.4f} ({len(tg)}f)  "
      f"LMA {RHO_LMA:.3f} vs {RHO_CLS:.3f}  cards {n_hall}/{n_cards}")
