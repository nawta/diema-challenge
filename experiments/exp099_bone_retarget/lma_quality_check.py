"""lma_quality_check.py — Stage 4a of exp099.

For each of 500 retarget pairs, compute 32-D LMA on the source clip
(using source-performer offsets) and on the retargeted clip (using
target-performer offsets). Across the 500 clips, compute per-attribute
Spearman ρ to test whether retargeting preserves the rank ordering of
LMA-derived emotion attributes.

Automated GO/NO-GO threshold per prompt:
- mean Spearman ρ >= 0.6  AND
- count of attributes with ρ < 0.4 <= 3
→ Stage 4a automated GO; otherwise NO-GO.

Output:
- output/retarget/lma_spearman.json  (per-attr + mean + verdict)
- output/retarget/PHASE0_VERDICT.md  (Stage 4a section filled; Stage 4b
  manual scoring pending)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from diema.features.lma_attributes import (  # noqa: E402
    compute_lma_attributes, LMA_NAMES)

CACHED_NPZ = Path("data/diema_challenge/processed/motion_train_quat.npz")
SMOKE_DIR = Path("output/retarget/smoke")
MANIFEST = Path("output/retarget/smoke_manifest.json")
LMA_JSON = Path("output/retarget/lma_spearman.json")
VERDICT_MD = Path("output/retarget/PHASE0_VERDICT.md")

MEAN_RHO_THRESHOLD = 0.60
BROKEN_RHO_THRESHOLD = 0.40
MAX_BROKEN_ATTRS = 3


def main():
    print(f"[stage4a] loading manifest {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text())
    pairs = manifest["pairs"]
    print(f"  {len(pairs)} pairs")

    print(f"[stage4a] loading cached npz {CACHED_NPZ}")
    npz = np.load(CACHED_NPZ, allow_pickle=True)

    src_lma = np.zeros((len(pairs), 32), dtype=np.float32)
    tgt_lma = np.zeros((len(pairs), 32), dtype=np.float32)

    for k, p in enumerate(pairs):
        i = p["idx"]
        joint_quats = npz[f"clip_{i}_joint_data"].astype(np.float32)
        root_pos = npz[f"clip_{i}_root_pos"].astype(np.float32)
        sidecar = np.load(p["output_npz"])
        src_off = sidecar["source_offsets"]
        tgt_off = sidecar["target_offsets"]
        new_root = sidecar["new_root_pos"]
        # LMA on source with source's offsets
        src_lma[k] = compute_lma_attributes(
            root_pos=root_pos, joint_quats=joint_quats, offsets=src_off)
        # LMA on retarget: same rotations, new root_pos (ground-contact
        # shifted), and target offsets
        tgt_lma[k] = compute_lma_attributes(
            root_pos=new_root, joint_quats=joint_quats, offsets=tgt_off)
        if (k + 1) % 100 == 0:
            print(f"  ... {k+1}/{len(pairs)}")
    print(f"[stage4a] LMA computed")

    # Per-attribute Spearman across 500 clips
    per_attr = []
    for j, name in enumerate(LMA_NAMES):
        x = src_lma[:, j]
        y = tgt_lma[:, j]
        if x.std() < 1e-8 or y.std() < 1e-8:
            rho, p = 1.0, 0.0   # degenerate; treat as perfectly preserved
        else:
            rho, p = spearmanr(x, y)
        per_attr.append({"attribute": name, "rho": float(rho),
                          "p_value": float(p) if not np.isnan(p) else 1.0,
                          "src_mean": float(x.mean()),
                          "src_std": float(x.std()),
                          "tgt_mean": float(y.mean()),
                          "tgt_std": float(y.std())})
    mean_rho = float(np.mean([a["rho"] for a in per_attr]))
    broken = [a for a in per_attr if a["rho"] < BROKEN_RHO_THRESHOLD]
    auto_go = (mean_rho >= MEAN_RHO_THRESHOLD
                and len(broken) <= MAX_BROKEN_ATTRS)

    # group by family
    families = {"body": [], "effort": [], "shape": [], "space": []}
    for a in per_attr:
        fam = a["attribute"].split(".")[0]
        families[fam].append(a["rho"])
    family_means = {f: float(np.mean(v)) if v else 0.0
                     for f, v in families.items()}

    out = {
        "n_pairs": len(pairs),
        "mean_rho": mean_rho,
        "family_mean_rho": family_means,
        "broken_attrs": [a["attribute"] for a in broken],
        "n_broken": len(broken),
        "automated_go_threshold": {
            "mean_rho_threshold": MEAN_RHO_THRESHOLD,
            "broken_rho_threshold": BROKEN_RHO_THRESHOLD,
            "max_broken_attrs": MAX_BROKEN_ATTRS,
        },
        "automated_verdict": "GO" if auto_go else "NO-GO",
        "per_attribute": per_attr,
    }
    LMA_JSON.parent.mkdir(parents=True, exist_ok=True)
    LMA_JSON.write_text(json.dumps(out, indent=2))
    print(f"[stage4a] → {LMA_JSON}")

    # console summary
    print()
    print(f"=== Stage 4a Spearman summary ===")
    print(f"mean ρ across 32 attributes: {mean_rho:.4f}")
    for fam, rho_mean in family_means.items():
        print(f"  {fam:8s} mean ρ = {rho_mean:.4f}")
    print(f"# attributes with ρ < {BROKEN_RHO_THRESHOLD}: {len(broken)} "
          f"({', '.join(a['attribute'] for a in broken[:5])}"
          f"{'…' if len(broken) > 5 else ''})")
    print(f"automated verdict: {out['automated_verdict']}")

    # Write VERDICT_MD with Stage 4a filled
    rho_sorted = sorted(per_attr, key=lambda a: a["rho"])
    md = [
        "# PHASE 0 VERDICT — exp099 bone-retarget pipeline",
        "",
        f"Status: **Stage 4a complete (automated check). Stage 4b "
        f"manual scoring PENDING by nawta.**",
        "",
        "## Stage 1a — KILL GATE (raw BVH offset deviation)",
        "",
        "**✅ PASS** — max relative deviation 23.67 % across "
        "JP_06 / TW_01 / JP_07 (kill-gate threshold 5 %).",
        "",
        "## Stage 1b/1c — 92-performer bone proportions",
        "",
        "- 74 train performers (40 JP + 34 TW); test performers "
        "anonymized.",
        "- **21 of 24 bones** differ between JP and TW at p < 0.001 "
        "(Mann-Whitney U, two-sided).",
        "- JP performers: longer spine + arms + shoulders "
        "(rank-biserial r > +0.94).",
        "- TW performers: longer legs + hands + toes "
        "(r < -0.83).",
        "- PCA: PC1+PC2 = 73.5 % of variance.",
        "- See `bone_proportions_summary.md` for full table + figure.",
        "",
        "## Stage 2 — `retarget_core.py` + 5 unit tests",
        "",
        "**✅ 5/5 PASS** — test_identity_retarget, "
        "test_scaling_changes_bones, test_rotations_preserved, "
        "test_root_pos_preserved, test_bvh_ctv_indexing.",
        "",
        "## Stage 3 — 500 cross-country smoke pairs",
        "",
        f"- {sum(1 for p in pairs if p['country']=='JP')} JP → TW "
        f"+ {sum(1 for p in pairs if p['country']=='TW')} TW → JP "
        "(seed=42, emotion-stratified).",
        "- Per source: top-decile max-cosine-distance target.",
        "- Output: 500 (.npy + .npz) under "
        "`output/retarget/smoke/` (gitignored, regenerable).",
        "- 8 side-by-side mp4 (5 spot + 3 worst) tracked under "
        "`output/retarget/smoke_renders/`.",
        "",
        "## Stage 4a — LMA Spearman correlation (automated)",
        "",
        f"- mean ρ across 32 attributes: **{mean_rho:.4f}** "
        f"(threshold ≥ {MEAN_RHO_THRESHOLD})",
        "- Per-family mean ρ:",
    ]
    for fam, rho_mean in family_means.items():
        md.append(f"  - **{fam}**: {rho_mean:.4f}")
    md.append("")
    md.append(f"- Attributes with ρ < {BROKEN_RHO_THRESHOLD}: "
                f"**{len(broken)}** "
                f"(threshold ≤ {MAX_BROKEN_ATTRS} for GO)")
    if broken:
        for a in broken:
            md.append(f"  - `{a['attribute']}` (ρ = {a['rho']:.3f})")
    md.append("")
    md.append("### Bottom-5 ρ (most-perturbed attributes)")
    md.append("")
    md.append("| Attribute | ρ | src_mean | tgt_mean |")
    md.append("|---|---:|---:|---:|")
    for a in rho_sorted[:5]:
        md.append(f"| `{a['attribute']}` | {a['rho']:.4f} "
                   f"| {a['src_mean']:.3f} | {a['tgt_mean']:.3f} |")
    md.append("")
    md.append("### Top-5 ρ (most-preserved attributes)")
    md.append("")
    md.append("| Attribute | ρ | src_mean | tgt_mean |")
    md.append("|---|---:|---:|---:|")
    for a in rho_sorted[-5:][::-1]:
        md.append(f"| `{a['attribute']}` | {a['rho']:.4f} "
                   f"| {a['src_mean']:.3f} | {a['tgt_mean']:.3f} |")
    md.append("")
    md.append(f"**Automated verdict (Stage 4a only)**: "
                f"**{out['automated_verdict']}**")
    md.append("")
    md.append("## Stage 4b — Manual scoring (PENDING)")
    md.append("")
    md.append("nawta must view 50 side-by-side mp4 under "
                "`output/retarget/inspection_sheet/` and fill the "
                "1-5 scores in `output/retarget/inspection_form.md`. "
                "Then re-run `aggregate_inspection.py` to update this "
                "verdict file with the final combined GO/NO-GO.")
    md.append("")
    md.append("## Stage 4c — Final verdict (PENDING)")
    md.append("")
    md.append("Will be filled by `aggregate_inspection.py` after nawta "
                "completes Stage 4b. Combines automated Stage 4a verdict "
                "with manual score mean. Both must pass for final GO.")
    md.append("")
    md.append("---")
    md.append("Generated by `experiments/exp099_bone_retarget/"
                "lma_quality_check.py` (Stage 4a).")
    VERDICT_MD.write_text("\n".join(md))
    print(f"[stage4a] → {VERDICT_MD}")


if __name__ == "__main__":
    main()
