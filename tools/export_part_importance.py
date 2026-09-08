""" export per-part projection-norm diagnostics to CSV + markdown.

See: (exp052 part importance)
Related:
- diema/evaluation/part_importance.py (pure-tensor compute)
- diema/models/conv1d_transformer/partalign_conv1d_tr_model.py (emits z_parts_raw)
- tools/eval_motion_text_retrieval.py (sibling consumer of the same dumped .pt)

Consumes a pre-dumped ``.pt`` file with::

    {
        "z_parts_raw": (N, P, D) float32   — pre-normalize projection
        "part_mask":   (N, P)    bool      — False = placeholder part
        "labels":      (N,)      int64     — emotion class id
        "stems":       list[str] (optional) — for provenance
        "part_names":  list[str] (optional, falls back to default 6-part order)
        "emotion_names": list[str] (optional, for readable labels)
    }

This schema is what the exp-specific val dumper writes post-training (the
same two-stage design used by eval_motion_text_retrieval).

Usage::

    python tools/export_part_importance.py \\
        --input /tmp/exp052_a00_fold00_val_parts.pt \\
        --output-csv docs/analysis/part_importance_exp052_a00_fold_00.csv \\
        --output-md  docs/analysis/part_importance_exp052_a00_fold_00.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diema.evaluation.part_importance import compute_part_importance  # noqa: E402

DEFAULT_PART_NAMES = (
    "head", "torso", "left_arm", "right_arm", "left_leg", "right_leg",
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(
            str(r.get(c, ""))
            .replace(",", ";")
            .replace("\n", " ")
            for c in cols
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_markdown(stats: dict, source: Path) -> str:
    parts = stats["part_names"]
    md = [
        f"# Part importance report",
        "",
        f"- source: `{source}`",
        f"- parts: {len(parts)}",
        f"- emotions: {len(stats['emotion_names'])}",
        "",
        "## 1. Per-part projection norm (pre-normalize)",
        "",
        "Higher mean norm = the model routes more signal to that part overall.",
        "",
        "| part | n | mean ||z_p|| | std |",
        "|---|---|---|---|",
    ]
    for row in stats["norm_stats"]:
        md.append(
            f"| {row['part']} | {row['n']} | "
            f"{row['mean']:.4f} | {row['std']:.4f} |"
        )
    md.extend([
        "",
        "## 2. Per-(emotion, part) mean norm",
        "",
        "| emotion | " + " | ".join(parts) + " |",
        "|---|" + "|".join(["---"] * len(parts)) + "|",
    ])
    # Pivot per_emotion_norm into a matrix.
    by_emo_part: dict[str, dict[str, float]] = {}
    for r in stats["per_emotion_norm"]:
        by_emo_part.setdefault(r["emotion"], {})[r["part"]] = r["mean"]
    for emo in stats["emotion_names"]:
        md.append(
            f"| {emo} | "
            + " | ".join(f"{by_emo_part.get(emo, {}).get(p, 0.0):.3f}" for p in parts)
            + " |"
        )
    md.extend([
        "",
        "## 3. Per-(emotion, part) cosine to global mean (normalized space)",
        "",
        "Lower = that emotion's part signal is distinctive; 1.0 = identical to the pooled mean.",
        "",
        "| emotion | " + " | ".join(parts) + " |",
        "|---|" + "|".join(["---"] * len(parts)) + "|",
    ])
    by_emo_cos: dict[str, dict[str, float]] = {}
    for r in stats["per_emotion_cosine"]:
        by_emo_cos.setdefault(r["emotion"], {})[r["part"]] = r["cosine_to_global"]
    for emo in stats["emotion_names"]:
        md.append(
            f"| {emo} | "
            + " | ".join(f"{by_emo_cos.get(emo, {}).get(p, 1.0):.3f}" for p in parts)
            + " |"
        )
    md.append("")
    return "\n".join(md)


@click.command()
@click.option(
    "--input", "input_path", required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--output-csv", default=None, type=click.Path(dir_okay=False),
    help="Defaults to the markdown path with .csv suffix.",
)
@click.option(
    "--output-md", default=None, type=click.Path(dir_okay=False),
    help="Default: docs/analysis/part_importance_<input_stem>.md",
)
@click.option(
    "--output-json", default=None, type=click.Path(dir_okay=False),
    help="Optional full-stats JSON dump.",
)
def main(
    input_path: str,
    output_csv: str | None,
    output_md: str | None,
    output_json: str | None,
) -> None:
    ip = Path(input_path)
    payload = torch.load(ip, map_location="cpu", weights_only=False)
    for required in ("z_parts_raw", "part_mask", "labels"):
        if required not in payload:
            raise click.ClickException(
                f"{ip} missing required key {required!r}"
            )

    part_names = list(payload.get("part_names", DEFAULT_PART_NAMES))
    emotion_names = payload.get("emotion_names")

    stats = compute_part_importance(
        z_parts_raw=payload["z_parts_raw"],
        part_mask=payload["part_mask"],
        labels=payload["labels"],
        part_names=part_names,
        emotion_names=emotion_names,
    )

    out_md = Path(output_md) if output_md else Path(
        f"docs/analysis/part_importance_{ip.stem}.md"
    )
    out_csv = Path(output_csv) if output_csv else out_md.with_suffix(".csv")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    _write_csv(out_csv, stats["per_emotion_norm"])
    click.echo(f"Saved: {out_csv}")

    md_text = _render_markdown(stats, source=ip)
    out_md.write_text(md_text, encoding="utf-8")
    click.echo(f"Saved: {out_md}")

    if output_json:
        out_json = Path(output_json)
        out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        click.echo(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
