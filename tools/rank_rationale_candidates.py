""" / exp061 — score-and-select among 3 v2 rationale candidates.

See: / tools/generate_rationale_candidates_qwen.py
Related:
- tools/check_rationale_leakage.py (v1 leakage scanner — still authoritative
  for the corpus-level audit; this script reuses its forbidden-token YAML)
- tools/probe_rationale_text_classifiability_v2.py (downstream gate test)

For each ``(render_key)`` group of candidates ``__a/b/c.json`` produced by
``tools/generate_rationale_candidates_qwen.py``, computes a per-candidate
score and writes the top-1 plus alt-1 into
``output/rationale_cache_v2/selected/<rat_key_top>.json``. The output file
is structurally identical to a v1 rationale JSON (top-level ``rationale``
field), so it drops in unchanged for the probe / cache builder.

Scoring (no learned components; deterministic so the smoke run + full run
agree):

1. **Schema validity** (binary) — all 9 keys present, type-correct. Failure
   → candidate is rejected (score=0).
2. **Leakage** (binary) — uses ``configs/leakage/forbidden_tokens.yaml`` to
   block emotion / country / filename hits. Hit → score=0 even if schema is
   valid (paper-grade gate).
3. **Body-part completeness** ∈ [0, 1] — fraction of (head, torso, l_arm,
   r_arm, l_leg, r_leg) that contain non-placeholder text.
4. **Lexical specificity** ∈ [0, 1] — average part-text length, soft-clipped
   at 100 chars (≈ the v1 corpus average per-part).
5. **Type-token ratio** ∈ [0, 1] — unique-tokens / total-tokens across the
   7 free-text fields. Discourages repetitive / templated wording.
6. **Salient-interval coverage** ∈ [0, 1] — ``min(3, n_intervals) / 3``.

Combined::

    score = 0.40 * completeness
          + 0.30 * specificity
          + 0.15 * ttr
          + 0.15 * interval_score

Schema-fail and leakage-block override the score to 0 unconditionally.

Optional rerank (--rerank-with-vlm-embedding) is intentionally not enabled
by default — it requires loading Qwen3-VL-Embedding (~30GB) and would
double the wall-clock. Add it later if the deterministic ranker leaves
ties or low-confidence picks.

Usage::

    # Smoke (only operates on already-generated candidates, fast)
    python tools/rank_rationale_candidates.py \\
        --candidates-dir output/rationale_cache_v2/rationales \\
        --output-dir output/rationale_cache_v2/selected \\
        --report-md docs/analysis/rationale_v2_ranker_smoke.md

    # Full corpus
    python tools/rank_rationale_candidates.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import click
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SCHEMA_KEYS = (
    "global_motion", "head", "torso", "left_arm", "right_arm",
    "left_leg", "right_leg", "salient_intervals", "uncertainty_notes",
)
PART_KEYS = ("head", "torso", "left_arm", "right_arm", "left_leg", "right_leg")
STRING_KEYS = (
    "global_motion", "head", "torso", "left_arm", "right_arm",
    "left_leg", "right_leg", "uncertainty_notes",
)

SCORING_VERSION = "v2.0.0"
WEIGHTS = {
    "completeness": 0.40,
    "specificity": 0.30,
    "ttr": 0.15,
    "interval_score": 0.15,
}


def is_placeholder(text: str) -> bool:
    return bool(re.match(
        r"^\s*(no\s+distinctive\s+motion|none|n/?a|)\s*\.?\s*$",
        str(text), re.IGNORECASE,
    ))


def _compile_word_boundary(aliases: list[str]) -> re.Pattern[str]:
    uniq = sorted({a.strip() for a in aliases if a.strip()}, key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(re.escape(a) for a in uniq) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def load_leakage_filters(yaml_path: Path):
    with yaml_path.open("r", encoding="utf-8") as f:
        tokens = yaml.safe_load(f)
    return {
        "emotion": _compile_word_boundary(tokens["emotion_aliases"]),
        "country": _compile_word_boundary(tokens["country_aliases"]),
        "filename": [re.compile(p) for p in tokens["filename_regex"]],
    }


def gather_text(rationale: dict) -> str:
    parts: list[str] = []
    for k in STRING_KEYS:
        v = rationale.get(k, "")
        if isinstance(v, str):
            parts.append(v)
    for item in rationale.get("salient_intervals", []) or []:
        if isinstance(item, dict):
            note = item.get("note")
            if isinstance(note, str):
                parts.append(note)
    return "\n".join(parts)


def has_leakage(rationale: dict, filters) -> tuple[bool, dict[str, list[str]]]:
    text = gather_text(rationale)
    hits: dict[str, list[str]] = {"emotion": [], "country": [], "filename": []}
    for m in filters["emotion"].finditer(text):
        hits["emotion"].append(m.group(0))
    for m in filters["country"].finditer(text):
        hits["country"].append(m.group(0))
    for pat in filters["filename"]:
        for m in pat.finditer(text):
            hits["filename"].append(m.group(0))
    blocked = any(len(v) > 0 for v in hits.values())
    return blocked, hits


def schema_valid(rationale: dict) -> bool:
    if not isinstance(rationale, dict):
        return False
    for k in SCHEMA_KEYS:
        if k not in rationale:
            return False
    for k in STRING_KEYS:
        if not isinstance(rationale[k], str):
            return False
    if not isinstance(rationale.get("salient_intervals", []), list):
        return False
    return True


def score_candidate(
    rationale: dict, filters,
) -> tuple[float, dict, dict[str, list[str]]]:
    debug: dict[str, float] = {}
    if not schema_valid(rationale):
        return 0.0, {"schema_ok": 0.0}, {"emotion": [], "country": [], "filename": []}

    blocked, hits = has_leakage(rationale, filters)
    if blocked:
        return 0.0, {"schema_ok": 1.0, "leakage_blocked": 1.0}, hits

    parts_filled = sum(
        1 for p in PART_KEYS
        if not is_placeholder(rationale[p])
    )
    completeness = parts_filled / len(PART_KEYS)

    part_chars = [len(str(rationale[k])) for k in PART_KEYS]
    avg_chars = sum(part_chars) / len(part_chars)
    specificity = min(1.0, avg_chars / 100.0)

    full_text = " ".join(str(rationale[k]) for k in STRING_KEYS).lower()
    tokens = re.findall(r"[a-z]+", full_text)
    if tokens:
        ttr = len(set(tokens)) / len(tokens)
    else:
        ttr = 0.0

    intervals = rationale.get("salient_intervals", [])
    n_intervals = (
        min(3, len(intervals))
        if isinstance(intervals, list) else 0
    )
    interval_score = n_intervals / 3.0

    score = (
        WEIGHTS["completeness"] * completeness
        + WEIGHTS["specificity"] * specificity
        + WEIGHTS["ttr"] * ttr
        + WEIGHTS["interval_score"] * interval_score
    )
    debug.update({
        "schema_ok": 1.0,
        "leakage_blocked": 0.0,
        "completeness": completeness,
        "specificity": specificity,
        "avg_chars": float(avg_chars),
        "ttr": ttr,
        "n_intervals": float(n_intervals),
    })
    return score, debug, hits


@click.command()
@click.option("--candidates-dir", default="output/rationale_cache_v2/rationales",
              show_default=True)
@click.option("--output-dir", default="output/rationale_cache_v2/selected",
              show_default=True)
@click.option("--forbidden", default="configs/leakage/forbidden_tokens.yaml",
              show_default=True)
@click.option("--report-md", default="docs/analysis/rationale_v2_ranker_report.md",
              show_default=True)
@click.option("--require-min-candidates", default=1, type=int, show_default=True,
              help="Skip a clip if it has fewer than N non-rejected candidates.")
def main(
    candidates_dir: str, output_dir: str,
    forbidden: str, report_md: str,
    require_min_candidates: int,
) -> None:
    cands_dir = Path(candidates_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filters = load_leakage_filters(Path(forbidden))

    files = sorted(cands_dir.glob("*__*.json"))
    if not files:
        raise click.ClickException(f"no candidate files in {cands_dir}")
    click.echo(f"Loaded {len(files)} candidate JSONs from {cands_dir}")

    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            click.echo(f"[skip] cannot parse {p.name}: {exc}", err=True)
            continue
        render_key = data.get("render_cache_key")
        variant = data.get("variant") or p.stem.split("__")[-1]
        if not render_key or variant not in {"a", "b", "c"}:
            click.echo(f"[skip] {p.name}: bad render_cache_key/variant", err=True)
            continue
        grouped[render_key][variant] = data

    click.echo(f"Grouped into {len(grouped)} (render_key) groups.")

    n_written = 0
    n_skipped_low_candidates = 0
    n_all_blocked = 0
    variant_winner_counts: Counter = Counter()
    score_dist: list[float] = []
    candidate_score_dist: list[tuple[str, float]] = []
    leakage_block_count: Counter = Counter({"a": 0, "b": 0, "c": 0})
    schema_fail_count: Counter = Counter({"a": 0, "b": 0, "c": 0})

    for render_key, variants_for_clip in grouped.items():
        scored: list[tuple[str, float, dict, dict]] = []
        for variant, data in variants_for_clip.items():
            rationale = data.get("rationale") or {}
            score, debug, hits = score_candidate(rationale, filters)
            scored.append((variant, score, debug, hits))
            candidate_score_dist.append((variant, score))
            if debug.get("schema_ok", 0.0) < 1.0:
                schema_fail_count[variant] += 1
            if debug.get("leakage_blocked", 0.0) >= 1.0:
                leakage_block_count[variant] += 1

        live = [t for t in scored if t[1] > 0.0]
        if len(live) < require_min_candidates:
            if len(live) == 0 and len(scored) > 0:
                n_all_blocked += 1
            n_skipped_low_candidates += 1
            continue

        live.sort(key=lambda t: t[1], reverse=True)
        top_variant, top_score, top_debug, _ = live[0]
        alt_variant, alt_score, alt_debug = (
            (live[1][0], live[1][1], live[1][2]) if len(live) > 1 else (None, None, None)
        )

        top_data = variants_for_clip[top_variant]
        alt_data = variants_for_clip[alt_variant] if alt_variant else None

        out_path = out_dir / f"{top_data['rationale_cache_key']}.json"
        payload = {
            "rationale_cache_key": top_data["rationale_cache_key"],
            "render_cache_key": render_key,
            "prompt_version": top_data.get("prompt_version", "2.0"),
            "variant": top_variant,
            "model_version": top_data.get("model_version"),
            "generated_at": top_data.get("generated_at"),
            "rationale": top_data["rationale"],
            "ranker_meta": {
                "scoring_version": SCORING_VERSION,
                "weights": WEIGHTS,
                "selected_variant": top_variant,
                "selected_score": top_score,
                "selected_debug": top_debug,
                "alt_variant": alt_variant,
                "alt_score": alt_score,
                "alt_debug": alt_debug,
                "alt_rationale_cache_key": (
                    alt_data["rationale_cache_key"] if alt_data else None
                ),
                "candidate_scores": {v: s for v, s, _, _ in scored},
                "leakage_blocked": {
                    v: bool(d.get("leakage_blocked", 0.0) >= 1.0)
                    for v, _, d, _ in scored
                },
            },
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        n_written += 1
        variant_winner_counts[top_variant] += 1
        score_dist.append(top_score)

    click.echo(
        f"Wrote {n_written} selected rationales to {out_dir} "
        f"(skipped {n_skipped_low_candidates}; all-blocked: {n_all_blocked})"
    )
    click.echo(f"Variant winners: {dict(variant_winner_counts)}")
    if score_dist:
        import statistics as st
        click.echo(
            f"Selected-score distribution: mean={st.mean(score_dist):.3f}, "
            f"median={st.median(score_dist):.3f}, min={min(score_dist):.3f}, "
            f"max={max(score_dist):.3f}"
        )

    md = ["# exp061 v2 candidate ranker report",
          "",
          f"- candidates dir: `{cands_dir}`",
          f"- selected dir: `{out_dir}`",
          f"- scoring version: `{SCORING_VERSION}`",
          f"- weights: `{WEIGHTS}`",
          "",
          "## Coverage",
          "",
          f"- candidate files scanned: **{len(files)}**",
          f"- (render_key) groups: **{len(grouped)}**",
          f"- selected written: **{n_written}**",
          f"- groups skipped (no live candidate): **{n_skipped_low_candidates}**",
          f"- groups where ALL candidates blocked/failed: **{n_all_blocked}**",
          "",
          "## Per-variant rejection counts",
          "",
          "| variant | schema-fail | leakage-block |",
          "|---|---:|---:|"]
    for v in ("a", "b", "c"):
        md.append(
            f"| {v} | {schema_fail_count[v]} | {leakage_block_count[v]} |"
        )
    md.extend([
        "",
        "## Variant winner distribution",
        "",
        "| variant | n_winner |",
        "|---|---:|"])
    for v in ("a", "b", "c"):
        md.append(f"| {v} | {variant_winner_counts[v]} |")

    if score_dist:
        import statistics as st
        md.extend([
            "",
            "## Selected-score distribution",
            "",
            f"- n: **{len(score_dist)}**",
            f"- mean: **{st.mean(score_dist):.3f}**",
            f"- median: **{st.median(score_dist):.3f}**",
            f"- min: **{min(score_dist):.3f}**",
            f"- max: **{max(score_dist):.3f}**",
        ])
    if candidate_score_dist:
        per_variant: dict[str, list[float]] = defaultdict(list)
        for v, s in candidate_score_dist:
            per_variant[v].append(s)
        md.extend([
            "",
            "## Per-variant candidate-score distribution",
            "",
            "| variant | n | mean | median | min | max |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        import statistics as st
        for v in ("a", "b", "c"):
            arr = per_variant.get(v, [])
            if not arr:
                md.append(f"| {v} | 0 | — | — | — | — |")
                continue
            md.append(
                f"| {v} | {len(arr)} | {st.mean(arr):.3f} | {st.median(arr):.3f} "
                f"| {min(arr):.3f} | {max(arr):.3f} |"
            )

    Path(report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(report_md).write_text("\n".join(md), encoding="utf-8")
    click.echo(f"Saved report: {report_md}")


if __name__ == "__main__":
    main()
