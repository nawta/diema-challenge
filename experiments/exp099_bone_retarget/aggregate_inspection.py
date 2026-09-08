"""aggregate_inspection.py — Stage 4c of exp099.

After nawta has filled the `score` column in
`output/retarget/inspection_form.md` (1-5 per pair, 50 rows), this
script parses the markdown table, aggregates scores, and updates
`PHASE0_VERDICT.md` with the final combined verdict.

Combination rule (decision framework §2):
- Stage 4a automated GO + manual mean >= 4.0 → final **GO + full 4000
  corpus** (mix ratio 0.4)
- Stage 4a automated GO + manual mean >= 3.5 → final **GO + half 2000
  corpus** (mix ratio 0.25)
- Stage 4a NO-GO OR manual mean < 3.5 → final **NO-GO** (kill)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

FORM = Path("output/retarget/inspection_form.md")
META = Path("output/retarget/inspection_sheet/metadata.json")
LMA_JSON = Path("output/retarget/lma_spearman.json")
VERDICT_MD = Path("output/retarget/PHASE0_VERDICT.md")

GO_MIX_FULL = 4.0
GO_MIX_HALF = 3.5


def parse_scores(form_md: str) -> list[dict]:
    """Parse the markdown table rows; extract slot, score, notes."""
    lines = form_md.splitlines()
    in_table = False
    rows = []
    for line in lines:
        if line.startswith("|---") or line.startswith("| slot"):
            in_table = True
            continue
        if not line.startswith("|"):
            in_table = False
            continue
        if not in_table:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7:
            continue
        slot_str, _mp4, emotion, _pair, _bd, score_str, notes = cells[:7]
        try:
            slot = int(slot_str)
        except ValueError:
            continue
        score = None
        if score_str:
            m = re.match(r"^\s*([1-5])\s*$", score_str)
            if m:
                score = int(m.group(1))
            else:
                # allow floats / decimals if user writes 4.5 etc.
                try:
                    sf = float(score_str)
                    if 1.0 <= sf <= 5.0:
                        score = sf
                except ValueError:
                    pass
        rows.append({
            "slot": slot,
            "emotion": emotion,
            "score": score,
            "notes": notes,
        })
    return rows


def main():
    if not FORM.exists():
        print(f"[stage4c] {FORM} not found — Stage 4b must run first.")
        sys.exit(2)
    rows = parse_scores(FORM.read_text())
    filled = [r for r in rows if r["score"] is not None]
    if not filled:
        print(f"[stage4c] no scores filled in {FORM}.")
        print("Open it, fill in 1-5 per row, then re-run.")
        sys.exit(1)
    if len(filled) < len(rows):
        print(f"[stage4c] WARNING: only {len(filled)}/{len(rows)} rows scored.")

    scores = np.array([r["score"] for r in filled], dtype=np.float64)
    mean_s = float(scores.mean())
    median_s = float(np.median(scores))
    std_s = float(scores.std())
    per_emo = defaultdict(list)
    for r in filled:
        per_emo[r["emotion"]].append(r["score"])
    emo_means = {emo: float(np.mean(v)) for emo, v in per_emo.items()}

    # Read automated Stage 4a verdict
    lma = json.loads(LMA_JSON.read_text())
    auto = lma["automated_verdict"]   # "GO" or "NO-GO"
    auto_mean_rho = lma["mean_rho"]

    # Final verdict
    if auto == "GO" and mean_s >= GO_MIX_FULL:
        final = "GO — full 4000 corpus, mix ratio 0.4"
    elif auto == "GO" and mean_s >= GO_MIX_HALF:
        final = "GO — half 2000 corpus, mix ratio 0.25"
    else:
        if auto != "GO":
            final = f"NO-GO — Stage 4a failed (mean ρ = {auto_mean_rho:.3f})"
        else:
            final = (f"NO-GO — manual mean {mean_s:.2f} < "
                      f"{GO_MIX_HALF} threshold")

    summary = {
        "n_filled": len(filled),
        "n_total": len(rows),
        "mean_score": mean_s,
        "median_score": median_s,
        "std_score": std_s,
        "per_emotion_mean": emo_means,
        "automated_stage4a_verdict": auto,
        "automated_stage4a_mean_rho": auto_mean_rho,
        "final_verdict": final,
    }
    json.dump(summary, open(VERDICT_MD.with_suffix(".json"), "w"), indent=2)
    print(f"[stage4c] mean score = {mean_s:.2f} ± {std_s:.2f} "
          f"(median {median_s:.2f}, n_filled = {len(filled)})")
    print(f"[stage4c] final verdict: {final}")

    # Update VERDICT_MD: replace Stage 4b/4c sections
    md = VERDICT_MD.read_text()
    pre, _sep, _rest = md.partition("## Stage 4b — Manual scoring")
    new_sections = [
        "## Stage 4b — Manual scoring (nawta-filled)",
        "",
        f"- N scored: {len(filled)}/{len(rows)}",
        f"- Mean score: **{mean_s:.2f}** ± {std_s:.2f} (median "
        f"{median_s:.2f})",
        "- Per-emotion mean:",
    ]
    for emo in sorted(emo_means.keys()):
        new_sections.append(f"  - {emo}: {emo_means[emo]:.2f}")
    new_sections.append("")
    if any(r["notes"] for r in filled):
        new_sections.append("- Notes from filled rows:")
        for r in filled:
            if r["notes"].strip():
                new_sections.append(f"  - slot {r['slot']:02d}: "
                                      f"{r['notes']}")
        new_sections.append("")
    new_sections += [
        "## Stage 4c — Final verdict",
        "",
        f"**{final}**",
        "",
        f"Decision logic: Stage 4a automated = {auto} "
        f"(mean ρ = {auto_mean_rho:.4f}); manual mean = {mean_s:.2f} "
        f"vs thresholds (≥{GO_MIX_FULL} full, ≥{GO_MIX_HALF} half).",
        "",
        " next step:",
    ]
    if "GO" in final and "full" in final:
        new_sections.append(
            "- Use decision framework §1 row 1 (full 4000 corpus, "
            "mix=0.4) and §2 row 1 (no warmup). Refer to "
            " to "
            "compose the prompt.")
    elif "GO" in final and "half" in final:
        new_sections.append(
            "- Use decision framework §1 row 2 (2000 corpus + "
            "per-fold sanity gate) and §2 row 2 (mix=0.25 with warmup).")
    else:
        new_sections.append(
            "- Abort synthetic-data track; redirect budget to Track A "
            "backlog per prompt's '完了後の流れ' section.")
    new_sections.append("")
    new_sections.append("---")
    new_sections.append("Generated by `aggregate_inspection.py` "
                         "(Stage 4c).")
    VERDICT_MD.write_text(pre + "\n".join(new_sections))
    print(f"[stage4c] → {VERDICT_MD} updated")


if __name__ == "__main__":
    main()
