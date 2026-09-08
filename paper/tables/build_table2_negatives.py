"""paper/tables/build_table2_negatives.py — Table 2 negative findings (P10).

Design doc: docs/experiments.md §"避けるべきアプローチ" + docs/results_summary.md
§5 (every number verified against those docs, not copied from the prompt).

Reader-facing columns (2026-06-15 revision): Approach | Configuration |
Outcome (ΔF1) | Methodological lesson. Internal experiment IDs
(expNNN / Phase XX-Y / aNN variant tags) and internal verdict jargon
(NO-GO / SIG / NM) are intentionally kept OUT of the rendered table — the
methodological lesson, not the run label, is what the reader needs. Effect
sizes and CIs are retained. Causal language avoided: "associated with /
regresses", never "causes".

Main table = 8 curated rows; full catalog -> sup_table2_negatives_full.*.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # _tbl

from _tbl import emit  # noqa: E402

HDR = ["Approach", "Configuration", "Outcome (ΔF1)", "Methodological lesson"]
COL = "l p{2.6cm} p{2.7cm} p{4.9cm}"

# 8 curated rows (doc-verified). Outcome folds the effect size together with
# its significance/CI; internal run IDs are intentionally omitted.
MAIN = [
    ["Supervised contrastive head", "pairwise SupCon, λ=0.05, batch 128",
     "−11.53 pp, significant",
     "12 classes at batch 128 give too few positives per class, suggesting "
     "a per-class center loss may fit better at this scale"],
    ["Mixture-of-experts fusion", "K=4, hand-crafted gradient-free routing",
     "collapses to 6.55% F1",
     "a shared head with random init and 21-D routing is associated with "
     "backbone collapse, "
     "suggesting a feature adapter, learned routing, and zero-init are needed"],
    ["Scenario-text alignment", "global InfoNCE, 3-fold",
     "−1.42 pp",
     "the projection collapses to performer identity "
     "(cross-group retrieval 10.6%)"],
    ["Part-rationale alignment", "cosine text alignment, 3-fold",
     "−0.64 pp, CI [−1.30, −0.09]",
     "the text bottleneck is associated with an F1 ceiling of 9–12%; "
     "shortcut-breaking works but "
     "yields no F1 lift"],
    ["VLM zero-shot distillation", "12-class, a recent VLM, preliminary",
     "failed in preliminary zero-shot ranking",
     "the true class ranked last (12/12); a vision-language model did not "
     "infer emotion reliably from faceless, scene-free skeleton renderings"],
    ["Contact-marker late fusion", "C3D 22-D contact stats, 3-fold",
     "−0.59 pp",
     "the contact features carry a 69.8% country signal, so some performers "
     "go out of distribution under LPO, suggesting a domain-invariance "
     "objective is needed"],
    ["Multi-crop window inference", "24-setting window sweep",
     "−12 to −17 pp",
     "the model is trained on full-span subsamples, so compact windows are a "
     "distribution shift"],
    ["Post-hoc calibration", "nested-LPO, 7-way ensemble",
     "0 to −0.14 pp, not significant",
     "equal-weight argmax is already near-optimal; bias-style calibration "
     "did not exceed it in this search"],
]

# full catalog (supplementary) — doc-verified extras appended
FULL = MAIN + [
    ["Same-architecture stacking", "8-way (six models + two ConvTr variants)",
     "−0.29 pp",
     "diversity saturates when stacking variants of a single architecture"],
    ["Over-stacked SSL ensemble", "13-way, three-plus MAMP checkpoints",
     "−0.88 pp",
     "one SSL family over-stacked dilutes the production members; two MAMP "
     "checkpoints is the sweet spot"],
    ["Arithmetic-mean averaging", "probability-domain averaging",
     "−1.15 pp (forgone)",
     "the canonical convention is logit-mean (raw-logit averaging); the "
     "arithmetic alternative leaves −1.15 pp on the table"],
    ["PoseC3D heatmap 3D-CNN", "preliminary evaluation",
     "20.26% (below the 24% threshold)",
     "architectural novelty is not error-space orthogonality; measure the "
     "contribution in error space"],
    ["Joint-masking sparse model", "options A/B, 3-fold",
     "−0.33 pp (B); collapse (A)",
     "attribution-driven sparse models are not ensemble-additive (the 7-way "
     "already sees those joints)"],
    ["Joint-subset training", "top-N joints only, 3-fold",
     "top-4 19.83%, CI [17.67, 21.99]",
     "input-driven sparsity converges to the same negative as joint-masking"],
    ["Heavy regularisation, short horizon", "Conv1D+Transformer, 40 epochs",
     "−1.34 pp standalone, CI [−0.32, +0.35]",
     "heavy regularisation needs a long horizon; it under-fits at 40 epochs"],
    ["Focal loss", "single-model variant",
     "−0.48 F1 / −0.99 Acc",
     "the class-balanced DIEM-A does not benefit from focal loss"],
    ["SkateFormer with SGD (lr=0.2)", "single-model variant",
     "chance-level collapse",
     "transformer backbones require AdamW"],
    ["Joint dropout (0.1)", "single-model variant",
     "seed-dependent fold collapse",
     "some folds collapse under some seeds; not adoptable"],
    ["Long clip length (128)", "STGCN++, clip 128 vs 64",
     "regresses",
     "the emotion signal concentrates in a short temporal window"],
    ["Content/style dual head", "GRL + style loss, default weights",
     "−2.65 pp",
     "GRL and a style loss obstruct backbone learning at default weights"],
    ["Gradient temporal saliency", "explainability probe",
     "AUC gap +0.003 (≈0), not significant",
     "within-window frame importance is near-uniform (entropy: per-sample "
     "98.7%, class-mean 99.5% of log T); use the body-part axis instead"],
]

CAP = ("Approaches that did not improve the system, each paired with its "
       "methodological lesson (non-causal wording: associated with / "
       "regresses / does not improve). ΔF1 is the paired change versus "
       "the matched baseline. The full catalogue is in the supplementary "
       "material.")
NOTE = []

out = emit("table2_negatives", HDR, MAIN, colspec=COL, bold_last=False,
           notes=NOTE, caption=CAP)
outS = emit("sup_table2_negatives_full", HDR, FULL, colspec=COL,
            bold_last=False,
            notes=["Full negative-results catalogue (supplementary); the main "
                   "paper shows the curated eight (Table 2)."],
            caption="Supplementary: full catalogue of negative results "
                    "(extends Table 2).")

print("Table 2 (main 8 rows):", out)
print("Sup Table 2 (full %d rows):" % len(FULL), outS)
for r in MAIN:
    print(" |", r[0], "|", r[2])
assert len(MAIN) == 8, "P10 requires exactly 8 curated rows"
for r in FULL:
    assert all(str(c).strip() for c in r), f"row '{r[0]}' has an empty cell"
print(f"\nall {len(FULL)} rows complete (4 cells each) ✓")
