""" / exp061: per-clip diff across 3 v2 rationale candidates.

See: / tools/generate_rationale_candidates_qwen.py
Related:
- tools/rank_rationale_candidates.py (deterministic top-1 selector)

Quick qualitative inspection tool. For each clip with all 3 variants present,
prints a side-by-side body-part diff so a human can judge whether the
candidates differ meaningfully (which is the whole premise of running 3
candidates instead of 1). Also computes coarse diversity metrics:

- Jaccard similarity of token sets across (a, b), (a, c), (b, c) per part.
- Top-1 candidate length spread (max - min) per part.

If the 3 candidates are highly similar (Jaccard > 0.85 mean), the 3-candidate
ranker buys very little over single-candidate v1 and the full-corpus run
should be reconsidered.

Usage::

    python tools/diff_rationale_candidates.py \\
        --candidates-dir output/rationale_cache_v2/rationales \\
        --report-md docs/analysis/rationale_v2_candidate_diff.md \\
        --max-clips 30
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PART_KEYS = ("global_motion", "head", "torso", "left_arm", "right_arm",
             "left_leg", "right_leg")


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", str(text).lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


@click.command()
@click.option("--candidates-dir", default="output/rationale_cache_v2/rationales",
              show_default=True)
@click.option("--report-md", default="docs/analysis/rationale_v2_candidate_diff.md",
              show_default=True)
@click.option("--max-clips", default=20, type=int, show_default=True,
              help="Show full text diff for the first N clips. All clips "
                   "contribute to the corpus-level diversity metrics.")
def main(candidates_dir: str, report_md: str, max_clips: int) -> None:
    cands_dir = Path(candidates_dir)
    files = sorted(cands_dir.glob("*__*.json"))
    if not files:
        raise click.ClickException(f"no candidate files in {cands_dir}")

    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        render_key = data.get("render_cache_key")
        variant = data.get("variant") or p.stem.split("__")[-1]
        if render_key and variant in {"a", "b", "c"}:
            grouped[render_key][variant] = data

    full_groups = {k: v for k, v in grouped.items() if len(v) == 3}
    click.echo(f"groups with all 3 variants: {len(full_groups)} "
               f"/ total groups: {len(grouped)}")

    pairs = [("a", "b"), ("a", "c"), ("b", "c")]
    jacc_per_part: dict[str, dict[tuple[str, str], list[float]]] = {
        p: {pair: [] for pair in pairs} for p in PART_KEYS
    }
    length_spread_per_part: dict[str, list[int]] = {p: [] for p in PART_KEYS}

    for render_key, variants_for_clip in full_groups.items():
        rats = {v: variants_for_clip[v]["rationale"] for v in ("a", "b", "c")}
        for p in PART_KEYS:
            tokens = {v: tokenize(rats[v].get(p, "")) for v in ("a", "b", "c")}
            for pair in pairs:
                jacc_per_part[p][pair].append(jaccard(tokens[pair[0]], tokens[pair[1]]))
            lens = [len(str(rats[v].get(p, ""))) for v in ("a", "b", "c")]
            length_spread_per_part[p].append(max(lens) - min(lens))

    md: list[str] = ["# exp061 v2 candidate diff report", ""]
    md.append(f"- candidates dir: `{cands_dir}`")
    md.append(f"- groups with all 3 variants: **{len(full_groups)}**")
    md.append(f"- total groups: {len(grouped)}")
    md.append("")
    md.append("## Per-part Jaccard token-set similarity (mean across clips)")
    md.append("")
    md.append("| part | a vs b | a vs c | b vs c |")
    md.append("|---|---:|---:|---:|")
    overall_means: list[float] = []
    for p in PART_KEYS:
        row = [f"| {p} "]
        for pair in pairs:
            arr = jacc_per_part[p][pair]
            mean = sum(arr) / len(arr) if arr else 0.0
            overall_means.append(mean)
            row.append(f"| {mean:.3f} ")
        row.append("|")
        md.append("".join(row))
    md.append("")
    if overall_means:
        grand_mean = sum(overall_means) / len(overall_means)
        md.append(f"**Grand-mean Jaccard**: {grand_mean:.3f}")
        if grand_mean > 0.85:
            md.append("")
            md.append("**Diversity verdict: LOW** — candidates are >85% token-overlap "
                      "on average. The 3-candidate strategy buys very little; "
                      "single-candidate v1 likely captures the same ceiling. "
                      "Reconsider full corpus regen.")
        elif grand_mean > 0.65:
            md.append("")
            md.append("**Diversity verdict: MODERATE** — candidates differ "
                      "enough to give the ranker something to choose from, "
                      "but the 3 prompts are not generating substantially "
                      "different rationales.")
        else:
            md.append("")
            md.append("**Diversity verdict: HIGH** — candidates differ substantially. "
                      "Ranker selection will materially affect the chosen text.")

    md.append("")
    md.append("## Per-part text-length spread (chars; mean across clips)")
    md.append("")
    md.append("| part | mean spread | max spread |")
    md.append("|---|---:|---:|")
    for p in PART_KEYS:
        arr = length_spread_per_part[p]
        if arr:
            md.append(f"| {p} | {sum(arr)/len(arr):.1f} | {max(arr)} |")
        else:
            md.append(f"| {p} | — | — |")

    md.append("")
    md.append(f"## Sample clip diffs (first {max_clips} clips with 3 variants)")
    md.append("")
    sampled = list(full_groups.items())[:max_clips]
    for render_key, variants_for_clip in sampled:
        md.append(f"### `{render_key[:16]}…`")
        md.append("")
        for p in PART_KEYS:
            md.append(f"**{p}**")
            for v in ("a", "b", "c"):
                txt = variants_for_clip[v]["rationale"].get(p, "")
                md.append(f"- *{v}*: {str(txt)[:250]}")
            md.append("")

    Path(report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(report_md).write_text("\n".join(md), encoding="utf-8")
    click.echo(f"Wrote: {report_md}")


if __name__ == "__main__":
    main()
