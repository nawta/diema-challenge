""" / exp062 — rule-based DSL extractor (CLI).

See: / diema/features/qwen_dsl.py (extractor logic)
Related:
- tools/probe_dsl_classifiability.py (downstream LogReg probe)

Reads every rationale JSON under ``output/rationale_cache/rationales/`` (v1
default; pass ``--rationales-dir output/rationale_cache_v2/selected`` for v2)
and produces a per-clip DSL JSON in ``output/dsl_cache_rule/<rat_key>.json``.

Each output file::

    {
      "rationale_cache_key": "...",
      "render_cache_key": "...",
      "source_prompt_version": "1.0" | "2.0",
      "extractor": "rule_v1.0",
      "dsl": {
        "global": {"openness": "...", "verticality": "...", ...},
        "head":   {"action": "...", "direction": "...", ...},
        "torso":  {...},
        "left_arm": {...},
        "right_arm": {...},
        "left_leg": {...},
        "right_leg": {...}
      }
    }

Idempotent: skips files already on disk.

Usage::

    # Extract from v1 cache (~7644 rationales, <1 min CPU)
    python tools/extract_rule_dsl.py

    # Extract from v2 selected cache (after exp061 ranker)
    python tools/extract_rule_dsl.py \\
        --rationales-dir output/rationale_cache_v2/selected \\
        --output-dir output/dsl_cache_rule_v2

After extraction, run ``tools/probe_dsl_classifiability.py``.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diema.features.qwen_dsl import (  # noqa: E402
    DSL_COLUMNS, GLOBAL_ATTR_VALUES, PART_ATTR_VALUES, PART_NAMES,
    extract_dsl, flatten_dsl,
)


EXTRACTOR_VERSION = "rule_v1.0"


@click.command()
@click.option("--rationales-dir", default="output/rationale_cache/rationales",
              show_default=True)
@click.option("--output-dir", default="output/dsl_cache_rule", show_default=True)
@click.option("--report-md", default="docs/analysis/exp062_dsl_extract_report.md",
              show_default=True)
@click.option("--prompt-version", default="any", show_default=True,
              help='Filter by source rationale prompt_version. "any" / "" '
                   'accepts all.')
@click.option("--max-clips", default=0, type=int,
              help="If >0, only process the first N rationales (smoke test).")
def main(
    rationales_dir: str, output_dir: str, report_md: str,
    prompt_version: str, max_clips: int,
) -> None:
    rat_dir = Path(rationales_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pv_filter = None if prompt_version.strip().lower() in ("", "any") else prompt_version

    files = sorted(rat_dir.glob("*.json"))
    if max_clips > 0:
        files = files[:max_clips]

    n_total = 0
    n_written = 0
    n_skipped_existing = 0
    n_skipped_other = 0
    distrib: dict[str, Counter] = {col: Counter() for col in DSL_COLUMNS}

    for p in files:
        n_total += 1
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            n_skipped_other += 1
            continue
        pv = str(data.get("prompt_version", ""))
        if pv_filter is not None and pv != pv_filter:
            n_skipped_other += 1
            continue
        rationale = data.get("rationale") or {}
        if not isinstance(rationale, dict):
            n_skipped_other += 1
            continue
        rk = data.get("rationale_cache_key") or p.stem
        out_path = out_dir / f"{rk}.json"
        if out_path.exists():
            n_skipped_existing += 1
            # Still tally distribution for reporting
            try:
                cached = json.loads(out_path.read_text(encoding="utf-8"))
                flat_cached = flatten_dsl(cached.get("dsl", {}))
                for col, val in flat_cached.items():
                    distrib[col][val] += 1
            except (json.JSONDecodeError, OSError):
                pass
            continue

        dsl = extract_dsl(rationale)
        flat = flatten_dsl(dsl)
        for col, val in flat.items():
            distrib[col][val] += 1

        payload = {
            "rationale_cache_key": rk,
            "render_cache_key": data.get("render_cache_key"),
            "source_prompt_version": pv,
            "extractor": EXTRACTOR_VERSION,
            "dsl": dsl,
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        n_written += 1

    click.echo(
        f"Scanned {n_total}; wrote {n_written}; cached {n_skipped_existing}; "
        f"skipped {n_skipped_other}."
    )

    md = ["# exp062 rule-based DSL extractor report", ""]
    md.append(f"- rationales dir: `{rat_dir}`")
    md.append(f"- output dir: `{out_dir}`")
    md.append(f"- extractor: `{EXTRACTOR_VERSION}`")
    md.append(f"- prompt-version filter: `{prompt_version}`")
    md.append(f"- scanned: **{n_total}**, wrote: **{n_written}**, cached: {n_skipped_existing}")
    md.append("")
    md.append("## Per-attribute value distribution")
    md.append("")
    md.append("Empty cells = value never produced by the extractor.")
    md.append("")
    md.append("| column | n_total | most-common (count) | distribution |")
    md.append("|---|---:|---|---|")
    for col in DSL_COLUMNS:
        section, attr = col.split("__", 1)
        if section == "global":
            valspace = GLOBAL_ATTR_VALUES[attr]
        else:
            valspace = PART_ATTR_VALUES[attr]
        c = distrib[col]
        n_col = sum(c.values())
        if n_col == 0:
            md.append(f"| {col} | 0 | — | — |")
            continue
        most_common = c.most_common(1)[0]
        dist_str = ", ".join(
            f"{v}={c.get(v, 0)} ({100.0*c.get(v, 0)/n_col:.1f}%)"
            for v in valspace
        )
        md.append(
            f"| {col} | {n_col} | {most_common[0]}={most_common[1]} | {dist_str} |"
        )
    Path(report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(report_md).write_text("\n".join(md), encoding="utf-8")
    click.echo(f"Saved report: {report_md}")


if __name__ == "__main__":
    main()
