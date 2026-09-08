""" prep: build the part-wise rationale text embedding cache.

See: (exp052)
Related:
- tools/generate_part_rationales_qwenvl_vllm.py (rationale producer)
- diema/features/text_features.py::ScenarioTextEncoder (shared encoder)
- diema/features/rationale_text_cache.py (runtime loader)

Reads every rationale JSON under ``output/rationale_cache/rationales/`` and
produces a single ``.pt`` cache containing:

- ``embeddings``   : Tensor (N_clips, 6, D)  — sentence-transformer vectors
                     for (head, torso, left_arm, right_arm, left_leg, right_leg)
- ``part_mask``    : Tensor (N_clips, 6) bool — False when the rationale for
                     that part is empty / placeholder ("no distinctive motion"
                     -> mask=False, i.e. part contributes nothing to the loss).
- ``stem_to_idx``  : dict[filename_stem → int] — stable row index
- ``part_names``   : ["head", "torso", "left_arm", "right_arm", "left_leg", "right_leg"]
- ``meta``         : prompt_version / model_version / encoder / schema / built_at

Mapping ``rationale_cache_key → filename_stem`` uses the audit log
(``output/rationale_cache/audit_log.jsonl``) or, for redundancy, the render
index (``output/rationale_cache/renders/render_index.jsonl``) + the
``render_cache_key`` field embedded in each rationale JSON.

Usage::

    python tools/build_rationale_text_cache.py \\
        --rationales-dir output/rationale_cache/rationales \\
        --render-index output/rationale_cache/renders/render_index.jsonl \\
        --output output/rationale_text_cache.pt
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diema.features.text_features import ScenarioTextEncoder  # noqa: E402


PART_NAMES: tuple[str, ...] = (
    "head", "torso", "left_arm", "right_arm", "left_leg", "right_leg",
)

_PLACEHOLDER_RE = re.compile(
    r"^\s*(no\s+distinctive\s+motion|none|n/?a|)\s*\.?\s*$",
    re.IGNORECASE,
)


def is_placeholder(text: str) -> bool:
    """True when a body-part rationale carries no informative content.

    Treats ``"no distinctive motion"``, ``"none"``, ``"n/a"``, ``""`` (with
    optional trailing punctuation / whitespace) as placeholder text that
    should be masked out of the alignment loss. Prevents a degenerate
    attractor where the motion projection collapses to whatever vector
    encodes the placeholder sentence.
    """
    return bool(_PLACEHOLDER_RE.match(str(text)))


def load_render_index(render_index_path: Path) -> dict[str, str]:
    """Load ``render_cache_key → filename_stem`` from render_index.jsonl."""
    mapping: dict[str, str] = {}
    if not render_index_path.is_file():
        return mapping
    with render_index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ck = rec.get("cache_key")
            stem = rec.get("stem")
            if isinstance(ck, str) and isinstance(stem, str):
                mapping[ck] = stem
    return mapping


def collect_rationales(
    rationales_dir: Path, prompt_version: str | None,
) -> list[dict]:
    """Load all rationale JSONs under ``rationales_dir`` that match prompt_version.

    If ``prompt_version`` is None, all are accepted. Results are deterministic
    via sorted filename iteration.
    """
    out: list[dict] = []
    for p in sorted(rationales_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if prompt_version is not None and str(data.get("prompt_version")) != prompt_version:
            continue
        out.append(data)
    return out


def _extract_part_text(rationale: dict, part: str) -> str:
    v = rationale.get(part)
    return v.strip() if isinstance(v, str) else ""


def build_cache(
    rationales: list[dict],
    stem_lookup: dict[str, str],
    encoder: ScenarioTextEncoder,
) -> dict:
    """Encode the six body-part texts for each rationale.

    Returns a payload dict ready to be torch.save'd:

    - embeddings   : (N, 6, D) float32
    - part_mask    : (N, 6) bool
    - stem_to_idx  : dict[str, int]
    - part_names   : tuple (frozen order)
    - meta         : dict

    Skips rationales whose filename stem cannot be resolved (warns via print).
    Placeholder texts ("no distinctive motion", etc.) are encoded normally
    but marked ``part_mask=False`` so downstream losses ignore them.
    """
    rows: list[tuple[str, list[str], list[bool]]] = []
    dropped = 0
    for data in rationales:
        rat = data.get("rationale") or {}
        if not isinstance(rat, dict):
            dropped += 1
            continue
        render_key = data.get("render_cache_key")
        stem = stem_lookup.get(str(render_key)) if isinstance(render_key, str) else None
        if not stem:
            dropped += 1
            continue
        texts: list[str] = []
        mask_row: list[bool] = []
        for part in PART_NAMES:
            txt = _extract_part_text(rat, part)
            texts.append(txt)
            mask_row.append(not is_placeholder(txt))
        rows.append((stem, texts, mask_row))

    if not rows:
        raise click.ClickException(
            "No resolvable rationales found — check --rationales-dir / "
            "--render-index paths or prompt-version filter."
        )

    # Encode all (N * 6) strings at once; sentence-transformer batch path is efficient.
    flat_texts: list[str] = [t for _, texts, _ in rows for t in texts]
    flat_vecs = encoder.encode(flat_texts, normalize=False)  # (N*6, D)
    if flat_vecs.ndim != 2:
        raise RuntimeError(
            f"expected 2-D encoding, got shape {tuple(flat_vecs.shape)}"
        )
    D = int(flat_vecs.shape[1])
    N = len(rows)
    embeddings = flat_vecs.view(N, len(PART_NAMES), D).to(torch.float32).contiguous()

    part_mask = torch.tensor(
        [m for _, _, m in rows], dtype=torch.bool
    )  # (N, 6)
    stem_to_idx: dict[str, int] = {stem: i for i, (stem, _, _) in enumerate(rows)}

    return {
        "embeddings": embeddings,
        "part_mask": part_mask,
        "stem_to_idx": stem_to_idx,
        "part_names": list(PART_NAMES),
        "n_dropped": dropped,
    }


@click.command()
@click.option(
    "--rationales-dir", default="output/rationale_cache/rationales", show_default=True,
)
@click.option(
    "--render-index",
    default="output/rationale_cache/renders/render_index.jsonl",
    show_default=True,
)
@click.option(
    "--output", default="output/rationale_text_cache.pt", show_default=True,
)
@click.option(
    "--prompt-version", default="1.0", show_default=True,
    help="Only ingest rationales whose top-level prompt_version matches.",
)
@click.option(
    "--encoder-model",
    default="sentence-transformers/all-MiniLM-L6-v2", show_default=True,
)
@click.option("--device", default=None, help="cpu / cuda / cuda:0. Default: auto.")
def main(
    rationales_dir: str, render_index: str, output: str,
    prompt_version: str, encoder_model: str, device: str | None,
) -> None:
    rat_dir = Path(rationales_dir)
    if not rat_dir.is_dir():
        raise click.ClickException(f"rationales dir not found: {rat_dir}")
    stem_lookup = load_render_index(Path(render_index))
    if not stem_lookup:
        raise click.ClickException(
            f"render index empty or missing: {render_index}"
        )
    rationales = collect_rationales(rat_dir, prompt_version=prompt_version)
    if not rationales:
        raise click.ClickException(
            f"no rationales under {rat_dir} match prompt_version={prompt_version!r}"
        )

    click.echo(
        f"Encoding {len(rationales)} × {len(PART_NAMES)} part texts via {encoder_model} ..."
    )
    encoder = ScenarioTextEncoder(model_name=encoder_model, device=device)
    payload = build_cache(rationales, stem_lookup, encoder)

    payload["meta"] = {
        "prompt_version": prompt_version,
        "encoder_model": encoder_model,
        "schema_version": "1.0",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rationales_scanned": str(len(rationales)),
        "n_rows_written": str(int(payload["embeddings"].shape[0])),
        "n_dropped": str(int(payload["n_dropped"])),
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    click.echo(
        f"Saved: {out} "
        f"(N={payload['embeddings'].shape[0]}, D={payload['embeddings'].shape[2]}, "
        f"dropped={payload['n_dropped']})"
    )


if __name__ == "__main__":
    main()
