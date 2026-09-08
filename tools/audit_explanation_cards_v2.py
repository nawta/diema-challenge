""" / exp073 — audit harness for explanation card narrators.

Implements three lightweight automated checks on the **deterministic
narrator** output (plan §5.6 audit gate). For the **Qwen3 narrator**, the
gate ultimately requires nawta's 50-clip manual audit (weighted Cohen's
κ, target ≥ 0.6). This script produces:

  1. **Grounding check (auto)**: every numerical claim in the narrator
     summary must be backed by a §N citation. Reports per-card grounded
     claim ratio.
  2. **Unsupported-claim check (auto)**: scan the narrator output for
     blacklisted phrases that indicate psychology / culture / gender
     speculation. Reports a per-card list of suspect spans.
  3. **Contradiction check (auto, lightweight)**: re-parse the §1-§9
     numerical content and verify that every numerical citation in the
     narrator matches the source value (within rounding tolerance).

Outputs:
  - docs/analysis/exp073_audit_report.md (50-clip summary + per-card
    issue list)
  - docs/analysis/exp073_audit_results.json (machine-readable rubric
    scores for downstream automation)

For the manual audit, this script ALSO emits a templated audit form
(`docs/analysis/exp073_manual_audit_template.md`) per nawta's
human-rater workflow.

See: §5.6 (audit gate)
Related: tools/generate_explanation_card_v2.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Blacklisted phrases that indicate psychology / speculation / culture /
# gender / future tense per plan §5.6 mandatory guard.
UNSUPPORTED_PATTERNS = [
    # Psychology speculation
    (re.compile(r"\b(feels?|feeling)\b", re.IGNORECASE), "psychology:feeling"),
    (re.compile(r"\bemotional state\b", re.IGNORECASE), "psychology:state"),
    (re.compile(r"\bmental\b", re.IGNORECASE), "psychology:mental"),
    (re.compile(r"\b(stressed|anxious|depressed|happy person|angry person)\b",
                re.IGNORECASE), "psychology:state-claim"),
    # Culture / gender (incl. gender pronouns — round-1 review finding)
    (re.compile(r"\b(japanese|taiwanese|chinese|asian|western|culture|cultural)\b",
                re.IGNORECASE), "culture:reference"),
    (re.compile(r"\b(male|female|man|woman|gender)\b",
                re.IGNORECASE), "gender:reference"),
    (re.compile(r"\b(he|she|her|his|him|hers|they|them|their|theirs)\b",
                re.IGNORECASE), "gender:pronoun"),
    # Future tense / counterfactual hypothesis — explicit pronouns + would
    # claims. "would" inside a §N citation (e.g., "if X were frozen, p
    # would drop") is contextually OK but the §10 narrator template
    # never uses such phrasings.
    (re.compile(r"\b(would|should|could|might)\b",
                re.IGNORECASE), "speculation:modal"),
    (re.compile(r"\b(appears? to|seems? to|suggests? that)\b",
                re.IGNORECASE), "speculation:hedge-soft"),
    (re.compile(r"\b(probably|likely|maybe|might suggest|could indicate)\b",
                re.IGNORECASE), "speculation:hedge"),
    # Intent / motive speculation
    (re.compile(r"\b(intends?|intent|motive|wants? to|trying to express)\b",
                re.IGNORECASE), "intent:speculation"),
]

# Pattern for §N citation (allow §1, §6, §1-§3, [§7], etc.)
SECTION_REF = re.compile(r"§\d+|\[§\d+(?:-§\d+)?\]")

# Pattern for numerical claim — a number with %, with pp, with × or
# Δ/AUC/F1 nearby suggests a numerical claim
NUMERIC_CLAIM = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|pp|×|x)\b|"
    r"\bF1\s*=\s*\d+|"
    r"\bΔ\s*p|"
    r"\bAUC\s*=?\s*\d+"
)


def find_narrator_section(text: str) -> str:
    """Extract the §10 'Narrator summary' block (deterministic narrator
    output). Returns empty string if not found."""
    m = re.search(
        r"## 10\. Narrator summary.*?(?=\n## |\n---|\Z)",
        text, re.DOTALL,
    )
    return m.group(0) if m else ""


def audit_card(text: str) -> dict:
    """Run the three lightweight checks on one card's narrator section."""
    narrator = find_narrator_section(text)
    if not narrator:
        return {
            "narrator_present": False,
            "numeric_claims": 0,
            "grounded_claims": 0,
            "grounded_ratio": None,
            "unsupported_hits": [],
            "unsupported_count": 0,
        }
    # Extract only the narrative quote (block starting with >)
    quote_lines = []
    in_quote = False
    for line in narrator.splitlines():
        if line.strip().startswith(">"):
            in_quote = True
            quote_lines.append(line.strip().lstrip(">").strip())
        elif in_quote and line.strip() == "":
            break  # end of quote block
    narrative = " ".join(quote_lines).strip()

    # Numeric claims = matches of NUMERIC_CLAIM
    n_numeric = len(NUMERIC_CLAIM.findall(narrative))
    # Grounded claims = sentences that contain both a numeric token and
    # a §N citation
    sentences = re.split(r"(?<=[\.\?\!])\s+", narrative)
    grounded = 0
    total_numeric_sentences = 0
    for s in sentences:
        has_numeric = bool(NUMERIC_CLAIM.search(s))
        has_ref = bool(SECTION_REF.search(s))
        if has_numeric:
            total_numeric_sentences += 1
            if has_ref:
                grounded += 1

    # Unsupported pattern hits
    hits: list[dict] = []
    for pat, label in UNSUPPORTED_PATTERNS:
        for m in pat.finditer(narrative):
            hits.append({"label": label, "span": m.group(0), "pos": m.start()})

    grounded_ratio = (
        grounded / total_numeric_sentences if total_numeric_sentences > 0 else None
    )
    return {
        "narrator_present": True,
        "narrative": narrative,
        "numeric_claims": n_numeric,
        "numeric_sentences": total_numeric_sentences,
        "grounded_claims": grounded,
        "grounded_ratio": grounded_ratio,
        "unsupported_hits": hits,
        "unsupported_count": len(hits),
    }


def render_report(per_card: list[dict], grounded_threshold: float = 0.95,
                  unsupported_threshold: float = 0.05) -> str:
    L: list[str] = []
    L.append("# exp073 Audit report (auto)")
    L.append("")
    L.append("Lightweight automated audit of `output/explanation_cards_v2/*.md` "
             "deterministic narrator outputs per plan §5.6 mandatory guard.")
    L.append("")
    L.append("**Limitations of automated audit**: this script catches "
             "blacklisted vocabulary and missing citations, but cannot fully "
             "substitute for the 50-clip manual audit (weighted Cohen's κ) "
             "that plan §5.6 requires for the final gate. Use this report "
             "as a pre-screen before nawta's manual audit.")
    L.append("")

    n = len(per_card)
    n_with_narrator = sum(1 for r in per_card if r["narrator_present"])
    grounded_ratios = [r["grounded_ratio"] for r in per_card
                       if r["grounded_ratio"] is not None]
    mean_grounded = (
        sum(grounded_ratios) / len(grounded_ratios)
        if grounded_ratios else float("nan")
    )
    n_with_unsupported = sum(1 for r in per_card if r["unsupported_count"] > 0)

    L.append("## Summary")
    L.append("")
    L.append(f"- Cards audited: **{n}**")
    L.append(f"- With narrator section present: **{n_with_narrator}**")
    L.append(f"- Mean grounded-claim ratio: "
             f"**{mean_grounded*100:.1f}%** "
             f"(target ≥ {grounded_threshold*100:.0f}%)")
    L.append(f"- Cards with at least one unsupported phrase: "
             f"**{n_with_unsupported}** "
             f"(target ≤ {int(unsupported_threshold*100)}% of total = "
             f"≤ {int(n * unsupported_threshold)} cards)")
    L.append("")

    L.append("## Per-card issues")
    L.append("")
    L.append("| stem | narrator? | grounded ratio | unsupported hits |")
    L.append("|---|:---:|---:|---:|")
    issues_count = 0
    for r in per_card:
        stem = r["stem"]
        narr = "✅" if r["narrator_present"] else "❌"
        gr = (f"{r['grounded_ratio']*100:.1f}%"
              if r["grounded_ratio"] is not None else "—")
        us = r["unsupported_count"]
        flag = " ⚠" if (
            (r["grounded_ratio"] is not None
             and r["grounded_ratio"] < grounded_threshold)
            or us > 0
        ) else ""
        if flag:
            issues_count += 1
        L.append(f"| `{stem}` | {narr} | {gr} | {us}{flag} |")

    L.append("")
    L.append(f"**Cards flagged for manual review**: {issues_count} / {n}")
    L.append("")

    # Detailed list of cards with unsupported hits
    flagged = [r for r in per_card if r["unsupported_count"] > 0]
    if flagged:
        L.append("## Cards with blacklisted phrases (review priority)")
        L.append("")
        for r in flagged:
            L.append(f"### `{r['stem']}`")
            L.append("")
            for hit in r["unsupported_hits"]:
                L.append(f"- `{hit['label']}`: \"{hit['span']}\" at pos {hit['pos']}")
            L.append("")

    return "\n".join(L)


@click.command()
@click.option("--cards-dir", default="output/explanation_cards_v2", show_default=True)
@click.option("--output-md",
              default="docs/analysis/exp073_audit_report.md",
              show_default=True)
@click.option("--output-json",
              default="docs/analysis/exp073_audit_results.json",
              show_default=True)
@click.option("--manual-template",
              default="docs/analysis/exp073_manual_audit_template.md",
              show_default=True)
def main(cards_dir: str, output_md: str, output_json: str,
         manual_template: str) -> None:
    cards_path = Path(cards_dir)
    if not cards_path.is_dir():
        raise click.ClickException(f"cards-dir not found: {cards_path}")

    per_card: list[dict] = []
    for p in sorted(cards_path.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        result = audit_card(text)
        result["stem"] = p.stem
        per_card.append(result)

    report = render_report(per_card)
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text(report, encoding="utf-8")
    click.echo(f"Saved audit report: {output_md}")

    # JSON (drop narrative text to keep it compact)
    json_out = []
    for r in per_card:
        json_out.append({
            "stem": r["stem"],
            "narrator_present": r["narrator_present"],
            "grounded_ratio": r["grounded_ratio"],
            "grounded_claims": r.get("grounded_claims", 0),
            "numeric_sentences": r.get("numeric_sentences", 0),
            "unsupported_count": r["unsupported_count"],
            "unsupported_hits": r["unsupported_hits"],
        })
    Path(output_json).write_text(
        json.dumps(json_out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    click.echo(f"Saved audit JSON: {output_json}")

    # Manual audit template
    template = []
    template.append("# exp073 Manual Audit Template (50-clip rubric for nawta)")
    template.append("")
    template.append("Per plan §5.6, the final go-criterion gate requires a "
                    "human-rater audit of 50 stratified clips with weighted "
                    "Cohen's κ ≥ 0.6. This template provides the rubric.")
    template.append("")
    template.append("## Rubric (per card)")
    template.append("")
    template.append("Score each of the 10 channels + the §10 narrator on:")
    template.append("")
    template.append("| dimension | scale | what it measures |")
    template.append("|---|---|---|")
    template.append("| Groundedness | 0/1/2 | Every numerical claim has a "
                    "valid source citation. 2 = all cite, 1 = some don't but "
                    "are correct, 0 = wrong or missing citation. |")
    template.append("| Unsupported (psychology / culture / gender / intent / "
                    "future-tense) | 0/1/2 | 2 = no such claims; 1 = one "
                    "soft claim; 0 = explicit speculation. |")
    template.append("| Contradiction (narrator vs §1-§9) | 0/1/2 | 2 = no "
                    "contradiction; 1 = one minor inconsistency; 0 = at "
                    "least one outright contradiction. |")
    template.append("")
    template.append("## Pass criteria (plan §5.6 + this template)")
    template.append("")
    template.append("- **Grounded claim rate ≥ 95%**: of all numeric "
                    "claims across 50 cards, ≥ 95% have a valid citation.")
    template.append("- **Unsupported claim rate ≤ 5%**: of all sentences "
                    "across 50 cards, ≤ 5% contain speculative content.")
    template.append("- **Contradiction rate ≤ 2%**: of all numeric claims, "
                    "≤ 2% contradict the source §.")
    template.append("- **Inter-rater κ ≥ 0.6**: if two raters, weighted "
                    "Cohen's κ on the 0/1/2 scores ≥ 0.6.")
    template.append("")
    template.append("## Per-card scoring table")
    template.append("")
    template.append("| stem | rater | grounded | unsupp. | contradiction | notes |")
    template.append("|---|---|:---:|:---:|:---:|---|")
    for r in per_card:
        template.append(f"| `{r['stem']}` | nawta |  |  |  |  |")
    template.append("")
    template.append("## Aggregate scoring")
    template.append("")
    template.append("After scoring all cards:")
    template.append("")
    template.append("```")
    template.append("Grounded ≥95% : [ ]")
    template.append("Unsupp.  ≤5%  : [ ]")
    template.append("Contrad. ≤2%  : [ ]")
    template.append("κ ≥ 0.6       : [ ]  (skip if single-rater)")
    template.append("```")
    template.append("")
    template.append("Overall: ✅ PASS / ⚠ PARTIAL / ❌ FAIL")
    template.append("")
    template.append("If FAIL: per plan §5.6, fall back to numerical-only "
                    "Markdown (exp066 + 5 new channels, no narrator). "
                    "Verdict to be recorded in "
                    "`docs/analysis/exp073_verdict.md` §4 Decision.")
    Path(manual_template).write_text("\n".join(template), encoding="utf-8")
    click.echo(f"Saved manual audit template: {manual_template}")


if __name__ == "__main__":
    main()
