""" deterministic 200-clip audit sampler.

See: / docs/cards/audit_sampling_protocol_3AD-7_v1.0.md
Related:
- tools/generate_part_rationales_qwenvl_vllm.py (produces rationales that this
  tool audits — token_cap_hit + duration_sec come from its audit log + mp4
  sidecar metadata)
- tools/check_rationale_leakage.py (corpus-wide regex audit; complementary)

Produces a reproducible JSONL manifest listing exactly 200 clips drawn from
the DIEM-A train split. Composition follows the 2-stage disproportional
stratified design:

  - 144 clips: baseline, 2 per (emotion × nationality × intensity) stratum
  - 56 clips: high-risk overlay on intensity-H strata
    - ~24 slots for HR-H   = "intensity extreme"
    - ~32 slots for HR-legs = "legs-placeholder-high"

The concrete per-cell count table ("overlay") is embedded below and matches
``docs/cards/audit_sampling_protocol_3AD-7_v1.0.md §3``. Total is exactly 200.

Output JSONL schema::

    {"stem": "JP_06_anger_1_H", "emotion": "anger", "nationality": "JP",
     "intensity": "H", "layer": "baseline" | "high_risk_intensity_extreme" |
     "high_risk_legs_placeholder_high"}

Deterministic via ``--seed 20260424`` (same default that seeded the Codex
allocation proposal).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASELINE_PER_CELL = 2

# Per-cell overlay counts on intensity-H strata only (mirrors the
# audit_sampling_protocol_3AD-7_v1.0.md §3 "High-risk overlay" table).
_OVERLAY: dict[tuple[str, str], int] = {
    ("anger", "JP"): 3, ("anger", "TW"): 2,
    ("contempt", "JP"): 2, ("contempt", "TW"): 2,
    ("disgust", "JP"): 3, ("disgust", "TW"): 2,
    ("fear", "JP"): 3, ("fear", "TW"): 2,
    ("gratitude", "JP"): 2, ("gratitude", "TW"): 2,
    ("guilt", "JP"): 2, ("guilt", "TW"): 2,
    ("jealousy", "JP"): 2, ("jealousy", "TW"): 2,
    ("joy", "JP"): 2, ("joy", "TW"): 3,
    ("pride", "JP"): 3, ("pride", "TW"): 2,
    ("sadness", "JP"): 3, ("sadness", "TW"): 2,
    ("shame", "JP"): 3, ("shame", "TW"): 2,
    ("surprise", "JP"): 2, ("surprise", "TW"): 3,
}

# Reason composition: (n_HR-H, n_HR-legs). Keeps Codex's per-cell balance.
_OVERLAY_COMPOSITION: dict[tuple[str, str], tuple[int, int]] = {
    ("anger", "JP"): (1, 2), ("anger", "TW"): (1, 1),
    ("contempt", "JP"): (1, 1), ("contempt", "TW"): (1, 1),
    ("disgust", "JP"): (1, 2), ("disgust", "TW"): (1, 1),
    ("fear", "JP"): (1, 2), ("fear", "TW"): (1, 1),
    ("gratitude", "JP"): (1, 1), ("gratitude", "TW"): (1, 1),
    ("guilt", "JP"): (1, 1), ("guilt", "TW"): (1, 1),
    ("jealousy", "JP"): (1, 1), ("jealousy", "TW"): (1, 1),
    ("joy", "JP"): (1, 1), ("joy", "TW"): (1, 2),
    ("pride", "JP"): (1, 2), ("pride", "TW"): (1, 1),
    ("sadness", "JP"): (1, 2), ("sadness", "TW"): (1, 1),
    ("shame", "JP"): (1, 2), ("shame", "TW"): (1, 1),
    ("surprise", "JP"): (1, 1), ("surprise", "TW"): (1, 2),
}


def parse_stem(stem: str) -> dict[str, str] | None:
    """Split ``JP_06_anger_1_H`` → {nationality, performer, emotion, intensity}."""
    parts = stem.split("_")
    if len(parts) < 5:
        return None
    return {
        "stem": stem,
        "nationality": parts[0],
        "performer": f"{parts[0]}_{parts[1]}",
        "emotion": parts[2],
        "intensity": parts[-1],
    }


def load_manifest(
    render_index: Path,
    risk_proxies: Path | None = None,
) -> list[dict]:
    """Load the candidate clip list from the render index.

    Each record = stem-derived fields + zero-valued risk proxies. If
    ``risk_proxies`` (a JSONL where each line has ``{stem, token_cap_hit,
    leg_motion_proxy, duration_sec}``) is supplied, merge those values in.
    """
    records: list[dict] = []
    with render_index.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            parsed = parse_stem(str(row.get("stem", "")))
            if parsed is None:
                continue
            parsed["token_cap_hit"] = 0
            parsed["leg_motion_proxy"] = 0.0
            parsed["duration_sec"] = 6.4        # default for 64-frame @ 10fps
            records.append(parsed)

    if risk_proxies is not None and risk_proxies.is_file():
        proxies: dict[str, dict] = {}
        with risk_proxies.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    stem = r.get("stem")
                    if stem:
                        proxies[str(stem)] = r
                except json.JSONDecodeError:
                    continue
        for rec in records:
            p = proxies.get(rec["stem"])
            if p:
                rec["token_cap_hit"] = int(p.get("token_cap_hit", 0))
                rec["leg_motion_proxy"] = float(p.get("leg_motion_proxy", 0))
                rec["duration_sec"] = float(p.get("duration_sec", rec["duration_sec"]))
    return records


def _group_by_cell(records: list[dict]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in records:
        key = (r["emotion"], r["nationality"], r["intensity"])
        out.setdefault(key, []).append(r)
    return out


def _sort_by_risk(records: list[dict]) -> list[dict]:
    """Rank candidates for the HR overlay — higher = more audit-worthy."""
    return sorted(
        records,
        key=lambda x: (
            -int(x["token_cap_hit"]),
            -float(x["leg_motion_proxy"]),
            -float(x["duration_sec"]),
            x["stem"],  # stable tie-break
        ),
    )


def pick_baseline_and_overlay(
    records: list[dict], *, seed: int,
) -> list[dict]:
    """Perform the 2-stage pick. Assumes every cell has ≥ BASELINE_PER_CELL
    clips and every intensity-H cell has ≥ BASELINE_PER_CELL + overlay[k]
    clips. In DIEM-A train (666/emotion, 74 performers) that is always the
    case.
    """
    rng = random.Random(seed)
    by_cell = _group_by_cell(records)
    chosen_ids: set[str] = set()
    out: list[dict] = []

    # -- Pass 1: baseline 2 per cell -----------------------------------
    for cell, items in sorted(by_cell.items()):
        shuffled = sorted(items, key=lambda x: x["stem"])
        rng.shuffle(shuffled)
        # Prefer distinct performers for coverage.
        selected: list[dict] = []
        used_performers: set[str] = set()
        for r in shuffled:
            if r["performer"] in used_performers:
                continue
            selected.append(r)
            used_performers.add(r["performer"])
            if len(selected) == BASELINE_PER_CELL:
                break
        # If performer-diversity constraint left us short, fall back to the
        # shuffled remainder.
        if len(selected) < BASELINE_PER_CELL:
            for r in shuffled:
                if r in selected:
                    continue
                selected.append(r)
                if len(selected) == BASELINE_PER_CELL:
                    break
        for r in selected:
            chosen_ids.add(r["stem"])
            out.append({
                "stem": r["stem"],
                "emotion": r["emotion"],
                "nationality": r["nationality"],
                "intensity": r["intensity"],
                "layer": "baseline",
            })

    # -- Pass 2: HR overlay (intensity H only) -------------------------
    for (emo, nat), n_slots in _OVERLAY.items():
        cell = (emo, nat, "H")
        pool = [r for r in by_cell.get(cell, []) if r["stem"] not in chosen_ids]
        ranked = _sort_by_risk(pool)
        n_hr_h, n_hr_legs = _OVERLAY_COMPOSITION[(emo, nat)]
        # Sanity: slot count must equal composition total.
        if n_slots != n_hr_h + n_hr_legs:
            raise AssertionError(
                f"overlay slots mismatch for {emo}/{nat}: "
                f"slots={n_slots}, composition={n_hr_h + n_hr_legs}"
            )
        if len(ranked) < n_slots:
            raise click.ClickException(
                f"not enough candidates for HR overlay in cell "
                f"{cell}: have {len(ranked)}, need {n_slots}"
            )
        for r in ranked[:n_hr_h]:
            out.append({
                "stem": r["stem"],
                "emotion": r["emotion"],
                "nationality": r["nationality"],
                "intensity": r["intensity"],
                "layer": "high_risk_intensity_extreme",
            })
        for r in ranked[n_hr_h : n_hr_h + n_hr_legs]:
            out.append({
                "stem": r["stem"],
                "emotion": r["emotion"],
                "nationality": r["nationality"],
                "intensity": r["intensity"],
                "layer": "high_risk_legs_placeholder_high",
            })

    return out


@click.command()
@click.option(
    "--render-index", default="output/rationale_cache/renders/render_index.jsonl",
    show_default=True,
)
@click.option(
    "--risk-proxies", default=None, type=click.Path(dir_okay=False),
    help="Optional JSONL with per-clip {stem, token_cap_hit, leg_motion_proxy, "
         "duration_sec}. Improves HR-overlay ranking; omit to use uniform ranks.",
)
@click.option(
    "--output", "output_path",
    default="docs/analysis/audit_sample_200_v1.jsonl", show_default=True,
)
@click.option("--seed", default=20260424, type=int, show_default=True)
def main(render_index: str, risk_proxies: str | None, output_path: str, seed: int) -> None:
    records = load_manifest(
        Path(render_index),
        Path(risk_proxies) if risk_proxies else None,
    )
    if not records:
        raise click.ClickException(f"render index empty: {render_index}")

    picks = pick_baseline_and_overlay(records, seed=seed)

    if len(picks) != 200:
        raise AssertionError(
            f"expected 200 picks, got {len(picks)} — check overlay table"
        )

    layer_counts = {
        "baseline": sum(1 for p in picks if p["layer"] == "baseline"),
        "high_risk_intensity_extreme": sum(
            1 for p in picks if p["layer"] == "high_risk_intensity_extreme"
        ),
        "high_risk_legs_placeholder_high": sum(
            1 for p in picks if p["layer"] == "high_risk_legs_placeholder_high"
        ),
    }
    if layer_counts["baseline"] != 144:
        raise AssertionError(
            f"baseline pass should be 144 clips, got {layer_counts['baseline']}"
        )
    if layer_counts["high_risk_intensity_extreme"] != 24:
        raise AssertionError(
            f"HR-H should be 24 clips, got {layer_counts['high_risk_intensity_extreme']}"
        )
    if layer_counts["high_risk_legs_placeholder_high"] != 32:
        raise AssertionError(
            f"HR-legs should be 32 clips, got {layer_counts['high_risk_legs_placeholder_high']}"
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in picks:
            f.write(json.dumps(row) + "\n")

    click.echo(f"Saved: {out}")
    click.echo(f"  total:                           {len(picks)}")
    click.echo(f"  baseline:                        {layer_counts['baseline']}")
    click.echo(f"  high_risk_intensity_extreme:     {layer_counts['high_risk_intensity_extreme']}")
    click.echo(f"  high_risk_legs_placeholder_high: {layer_counts['high_risk_legs_placeholder_high']}")


if __name__ == "__main__":
    main()
