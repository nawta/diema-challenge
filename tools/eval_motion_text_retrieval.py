""" (exp054): Motion ↔ Text retrieval evaluation CLI.

See: (exp054 retrieval probe)
Related:
- diema/evaluation/retrieval.py (tensor-level eval, covered by tests)
- diema/features/text_cache.py (ScenarioTextCache — --target scenario)
- diema/features/rationale_text_cache.py (RationaleTextCache — --target rationale)
- tools/dump_val_embeddings.py (produces the .pt consumed here) — exp-specific,
  ships alongside each exp's run.py (6 wrappers).

**Two-stage workflow** (so this tool stays exp-agnostic):

1. **Dump** ``z_motion`` + ``labels`` + ``stems`` for the val fold from an
   exp-specific script (the backbone instantiation cannot be generic —
   exp051 uses ``conv1d_transformer`` while exp052 uses the region-aware
   variant, and hparams do not unambiguously encode the backbone class).
2. **Probe** with this tool against a scenario or rationale cache.

Dumped ``.pt`` schema (dict)::

    {
        "z_motion": (N, D) float32 tensor — motion projection (L2-normalized recommended),
        "labels":   (N,)   int64 tensor — emotion class for same-class consistency,
        "stems":    list[str] length N — filename stems (used for positive lookup + groups),
        "target_kind": "scenario" | "rationale"  (optional),
        "exp_id":   free-form string (optional metadata),
    }

Usage::

    # scenario retrieval probe on pre-dumped exp051 embeddings
    python tools/eval_motion_text_retrieval.py \\
        --motion-embeddings /tmp/exp051_fold00_z_motion.pt \\
        --target scenario \\
        --output-md docs/analysis/retrieval_metrics_exp051_a00_fold_00.md

    # rationale retrieval probe on pre-dumped exp052 embeddings
    python tools/eval_motion_text_retrieval.py \\
        --motion-embeddings /tmp/exp052_fold00_z_parts_mean.pt \\
        --target rationale \\
        --output-md docs/analysis/retrieval_metrics_exp052_a00_fold_00.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _country_from_stem(stem: str) -> str:
    """Filename stems look like ``JP_06_anger_1_H`` or ``TW_05_joy_2_M``."""
    return stem.split("_", 1)[0] if "_" in stem else "UNK"


def _performer_group_from_stem(stem: str) -> str:
    """Performer id is ``<country>_<id>`` (first two tokens)."""
    parts = stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else stem


def _load_embeddings(path: Path) -> dict:
    if not path.is_file():
        raise click.ClickException(f"motion embeddings not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for required in ("z_motion", "labels", "stems"):
        if required not in payload:
            raise click.ClickException(
                f"{path} missing required key {required!r}; expected "
                "{z_motion, labels, stems}"
            )
    return payload


def build_scenario_targets(
    stems: list[str], scenario_cache_path: Path, train_csv: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(t_text (U, D), positive (N,))``. ``positive[i] = -1`` when unknown."""
    from diema.features.text_cache import ScenarioTextCache
    cache = ScenarioTextCache(
        cache_path=str(scenario_cache_path), csv_path=str(train_csv),
    )
    t = cache.embedding_matrix.to(torch.float32)
    pos = cache.idx_for_batch([Path(s).stem for s in stems])
    return t, pos


def build_rationale_targets(
    stems: list[str], rationale_cache_path: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool the 6-part text vectors into a single per-clip target.

    Mask is honored: placeholder parts are zeroed before averaging so the
    mean reflects only informative parts. Empty rows (all parts placeholder)
    fall back to an arbitrary unit vector but their positive index still
    maps to the row — callers can treat them as "no rationale" via mask.
    """
    from diema.features.rationale_text_cache import RationaleTextCache
    rc = RationaleTextCache(str(rationale_cache_path))
    mask = rc._part_mask.to(torch.float32).unsqueeze(-1)          # (N, P, 1)
    numerator = (rc._embeddings * mask).sum(dim=1)                 # (N, D)
    denom = mask.sum(dim=1).clamp_min(1e-12)                       # (N, 1)
    t = torch.nn.functional.normalize(numerator / denom, dim=1)
    pos = torch.tensor(
        [rc.idx_for(Path(s).stem) for s in stems], dtype=torch.long,
    )
    return t, pos


@click.command()
@click.option(
    "--motion-embeddings", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a .pt emitted by the exp-specific dumper (see module docstring).",
)
@click.option(
    "--target", type=click.Choice(["scenario", "rationale"]), required=True,
)
@click.option(
    "--scenario-cache", default="output/scenario_text_cache_masked.pt",
    show_default=True,
)
@click.option(
    "--train-csv", default="data/diema_challenge/raw/train_data.csv",
    show_default=True,
)
@click.option(
    "--rationale-cache", default="output/rationale_text_cache.pt",
    show_default=True,
)
@click.option(
    "--output-json", default=None, type=click.Path(dir_okay=False),
)
@click.option(
    "--output-md", default=None, type=click.Path(dir_okay=False),
)
def main(
    motion_embeddings: str, target: str,
    scenario_cache: str, train_csv: str, rationale_cache: str,
    output_json: str | None, output_md: str | None,
) -> None:
    from diema.evaluation.retrieval import motion_to_text_retrieval

    emb_path = Path(motion_embeddings)
    payload = _load_embeddings(emb_path)
    z_motion: torch.Tensor = payload["z_motion"].to(torch.float32)
    labels: torch.Tensor = payload["labels"].to(torch.long)
    stems: list[str] = list(payload["stems"])
    if z_motion.shape[0] != len(stems) or labels.shape[0] != len(stems):
        raise click.ClickException(
            "mismatched shapes: z_motion, labels, and stems must all agree on N"
        )

    if target == "scenario":
        t_text, positive = build_scenario_targets(
            stems, Path(scenario_cache), Path(train_csv),
        )
    else:
        t_text, positive = build_rationale_targets(stems, Path(rationale_cache))

    groups = torch.tensor(
        [hash(_performer_group_from_stem(Path(s).stem)) % (2**31) for s in stems],
        dtype=torch.long,
    )

    # Cap recall@k thresholds at the pool size so small synthetic / debug
    # runs don't error out. Real val folds will have M in the thousands.
    pool_size = int(t_text.shape[0])
    ks = tuple(k for k in (1, 5, 10) if k <= pool_size)
    if not ks:
        raise click.ClickException(
            f"pool size {pool_size} too small for Recall@1 — check cache"
        )

    metrics = motion_to_text_retrieval(
        z_motion, t_text, positive,
        ks=ks,
        labels=labels, groups=groups,
        same_class_k=min(5, pool_size), cross_group_k=min(5, pool_size),
    )

    country = [_country_from_stem(Path(s).stem) for s in stems]
    country_metrics: dict[str, dict[str, float]] = {}
    for c in sorted(set(country)):
        mask_c = torch.tensor([x == c for x in country])
        if mask_c.sum().item() < 2:
            continue
        country_metrics[c] = motion_to_text_retrieval(
            z_motion[mask_c], t_text, positive[mask_c],
            ks=ks,
            labels=labels[mask_c], groups=groups[mask_c],
            same_class_k=min(5, pool_size), cross_group_k=min(5, pool_size),
        )

    stem_id = emb_path.stem
    out_md = Path(output_md) if output_md else Path(
        f"docs/analysis/retrieval_metrics_{stem_id}.md"
    )
    out_json = Path(output_json) if output_json else out_md.with_suffix(".json")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    payload_out = {
        "motion_embeddings": str(emb_path),
        "target": target,
        "n_queries": int(len(stems)),
        "metrics": metrics,
        "per_country": country_metrics,
    }
    out_json.write_text(json.dumps(payload_out, indent=2), encoding="utf-8")
    click.echo(f"Saved: {out_json}")

    md = [
        f"# Retrieval metrics — {stem_id}",
        "",
        f"- motion embeddings: `{emb_path}`",
        f"- target: **{target}**",
        f"- queries: {len(stems)}",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for k, v in metrics.items():
        if isinstance(v, float):
            md.append(f"| {k} | {v:.4f} |")
        else:
            md.append(f"| {k} | {v} |")
    md.append("")
    if country_metrics:
        md.append("## Per-country")
        md.append("")
        keys = list(next(iter(country_metrics.values())).keys())
        md.append("| country | " + " | ".join(keys) + " |")
        md.append("|---|" + "|".join(["---"] * len(keys)) + "|")
        for c, m in country_metrics.items():
            md.append("| " + " | ".join(
                [c] + [
                    f"{m[k]:.4f}" if isinstance(m[k], float) else str(m[k])
                    for k in keys
                ]
            ) + " |")
        md.append("")
    out_md.write_text("\n".join(md), encoding="utf-8")
    click.echo(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
