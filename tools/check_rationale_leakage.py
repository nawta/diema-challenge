""" audit generated rationales for forbidden token leakage.

See: / configs/leakage/forbidden_tokens.yaml
Related:
- tools/generate_part_rationales_qwenvl.py (produces the rationale JSONs)
- docs/rule_and_ethics_checklist.md §4 (paper audit reporting)

Scans every rationale JSON in ``output/rationale_cache/rationales/`` (or a
specified dir) and checks the text fields against:

1. **emotion_aliases**   — word-boundary, case-insensitive
2. **country_aliases**    — word-boundary, case-insensitive
3. **filename_regex**     — raw regex (no IGNORECASE, preserves case-sensitive bands)
4. **hallucination_tokens** — word-boundary, case-insensitive (warning, not error)
5. **out_of_skeleton_tokens** — word-boundary, case-insensitive (warning)

Also emits a CSV + per-clip markdown trail so reviewers can spot-check.

Usage::

    python tools/check_rationale_leakage.py
    # or with a specific YAML / dir:
    python tools/check_rationale_leakage.py \\
        --forbidden configs/leakage/forbidden_tokens.yaml \\
        --rationales output/rationale_cache/rationales \\
        --output docs/analysis/rationale_leakage_report.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import click
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEXT_FIELDS = (
    "global_motion", "head", "torso",
    "left_arm", "right_arm", "left_leg", "right_leg",
    "uncertainty_notes",
)


def _compile_word_boundary(aliases: list[str]) -> re.Pattern[str]:
    uniq = sorted({a.strip() for a in aliases if a.strip()}, key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(re.escape(a) for a in uniq) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def _gather_text(rationale: dict) -> str:
    """Concatenate all text fields into one string for global scanning."""
    parts: list[str] = []
    for k in TEXT_FIELDS:
        v = rationale.get(k)
        if isinstance(v, str):
            parts.append(v)
    # Also include salient_intervals notes
    for item in rationale.get("salient_intervals", []) or []:
        if isinstance(item, dict):
            note = item.get("note")
            if isinstance(note, str):
                parts.append(note)
    return "\n".join(parts)


def scan_rationale(
    rationale: dict,
    emotion_regex: re.Pattern[str],
    country_regex: re.Pattern[str],
    filename_regexes: list[re.Pattern[str]],
    hallucination_regex: re.Pattern[str],
    oos_regex: re.Pattern[str],
) -> dict:
    """Return per-flag hit counts + sample snippets for one rationale."""
    text = _gather_text(rationale)
    hits: dict[str, list[str]] = {
        "emotion": [],
        "country": [],
        "filename_regex": [],
        "hallucination": [],
        "out_of_skeleton": [],
    }

    for m in emotion_regex.finditer(text):
        hits["emotion"].append(m.group(0))
    for m in country_regex.finditer(text):
        hits["country"].append(m.group(0))
    for pat in filename_regexes:
        for m in pat.finditer(text):
            hits["filename_regex"].append(m.group(0))
    for m in hallucination_regex.finditer(text):
        hits["hallucination"].append(m.group(0))
    for m in oos_regex.finditer(text):
        hits["out_of_skeleton"].append(m.group(0))

    total_block = len(hits["emotion"]) + len(hits["country"]) + len(hits["filename_regex"])
    total_warn = len(hits["hallucination"]) + len(hits["out_of_skeleton"])
    return {
        "hits": hits,
        "block_total": total_block,    # these force a rejection
        "warn_total": total_warn,      # flagged for manual review
        "text_chars": len(text),
    }


def load_forbidden(yaml_path: Path) -> dict:
    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--forbidden", default="configs/leakage/forbidden_tokens.yaml")
@click.option("--rationales", default="output/rationale_cache/rationales")
@click.option("--output-md", default="docs/analysis/rationale_leakage_report.md")
@click.option("--output-csv", default="docs/analysis/rationale_leakage_report.csv")
@click.option("--sample-snippet-len", default=80, type=int)
def main(
    forbidden: str,
    rationales: str,
    output_md: str,
    output_csv: str,
    sample_snippet_len: int,
) -> None:
    tokens = load_forbidden(Path(forbidden))
    emotion_regex = _compile_word_boundary(tokens["emotion_aliases"])
    country_regex = _compile_word_boundary(tokens["country_aliases"])
    filename_regexes = [re.compile(p) for p in tokens["filename_regex"]]
    hallucination_regex = _compile_word_boundary(tokens["hallucination_tokens"])
    oos_regex = _compile_word_boundary(tokens["out_of_skeleton_tokens"])

    rat_dir = Path(rationales)
    rat_files = sorted(rat_dir.glob("*.json"))
    if not rat_files:
        raise click.ClickException(f"no *.json in {rat_dir}")

    rows: list[dict] = []
    total_block = 0
    total_warn = 0
    for p in rat_files:
        data = json.loads(p.read_text())
        rationale = data.get("rationale", {})
        scan = scan_rationale(
            rationale, emotion_regex, country_regex, filename_regexes,
            hallucination_regex, oos_regex,
        )
        total_block += scan["block_total"]
        total_warn += scan["warn_total"]
        rows.append({
            "rationale_cache_key": data.get("rationale_cache_key", p.stem),
            **{f"hits_{k}": ",".join(v[:3]) for k, v in scan["hits"].items()},
            **{f"n_{k}": len(v) for k, v in scan["hits"].items()},
            "block_total": scan["block_total"],
            "warn_total": scan["warn_total"],
            "text_chars": scan["text_chars"],
        })

    # CSV
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    click.echo(f"Saved: {csv_path}")

    # Markdown summary
    n_files = len(rat_files)
    n_blocked = int((df["block_total"] > 0).sum())
    n_warned = int((df["warn_total"] > 0).sum())
    md: list[str] = [
        "# Rationale leakage audit ",
        "",
        f"- Rationales scanned: **{n_files}**",
        f"- Rationales with BLOCKING hits (emotion/country/filename): **{n_blocked}** ({100*n_blocked/max(n_files,1):.1f}%)",
        f"- Rationales with WARNING hits (hallucination/out-of-skeleton): **{n_warned}** ({100*n_warned/max(n_files,1):.1f}%)",
        f"- Total blocking hits across corpus: **{total_block}**",
        f"- Total warning hits across corpus: **{total_warn}**",
        "",
        "## By category (totals across all rationales)",
        "",
        "| category | total hits | unique tokens |",
        "|---|---|---|",
    ]
    categories = ["emotion", "country", "filename_regex", "hallucination", "out_of_skeleton"]
    for c in categories:
        col = f"n_{c}"
        total = int(df[col].sum())
        hits_col = f"hits_{c}"
        uniq_hits: set[str] = set()
        for hit_str in df[hits_col]:
            if isinstance(hit_str, str) and hit_str:
                uniq_hits.update(hit_str.split(","))
        md.append(f"| {c} | {total} | {len(uniq_hits)}: {sorted(uniq_hits)[:10]} |")

    md.extend([
        "",
        "## Block threshold",
        "",
        "A rationale is considered **blocked** (cannot be used for training) if",
        "any emotion_alias, country_alias, or filename_regex pattern matches its",
        "text fields. Warning-only rationales (hallucination / out_of_skeleton"
        " tokens) are kept but flagged for manual spot-check.",
        "",
        "## Top offenders (first 20 blocked rationales, if any)",
        "",
    ])
    blocked = df[df["block_total"] > 0].head(20)
    if len(blocked) == 0:
        md.append("_(none)_")
    else:
        md.append("| rationale_cache_key | emotion | country | filename_regex | sample hits |")
        md.append("|---|---|---|---|---|")
        for _, row in blocked.iterrows():
            md.append(
                f"| `{row['rationale_cache_key'][:12]}...` "
                f"| {row['n_emotion']} "
                f"| {row['n_country']} "
                f"| {row['n_filename_regex']} "
                f"| `{row['hits_emotion'] or row['hits_country'] or row['hits_filename_regex']}` |"
            )

    out_md = Path(output_md)
    out_md.write_text("\n".join(md))
    click.echo(f"Saved: {out_md}")

    click.echo(f"\nSummary: {n_blocked}/{n_files} blocked, {n_warned}/{n_files} warned.")
    raise SystemExit(0 if n_blocked == 0 else 1)


if __name__ == "__main__":
    main()
