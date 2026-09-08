""" strip train-time-only text/part weights from a checkpoint.

See: (exp053 Go → submission hygiene)
Related:
- diema/models/conv1d_transformer/textalign_conv1d_tr_model.py (text_head)
- diema/models/conv1d_transformer/partalign_conv1d_tr_model.py (part_head)
- diema/training/text_align_trainer.py (scenario_embedding_matrix buffer)
- diema/training/part_align_trainer.py (part_embedding_tensor / part_mask_tensor)
- tests/test_strip_text_branch.py (forward-equivalence + subset-key assertion)

Removes keys that ride in the checkpoint purely to support the train-time
alignment losses but are **never used during inference**:

- model.text_head.* — global scenario projection
- model.part_head.* — part-wise projection
- ``scenario_embedding_matrix`` — precomputed scenario text embeddings
- ``part_embedding_tensor``  — precomputed (N, P, D) rationale embeddings
- ``part_mask_tensor``       — matching placeholder mask

Leakage-adjacent rationale: keeping the part embedding tensor in a submitted
checkpoint could be construed as smuggling supervision text into the test
environment. Stripping is mandatory before submission even though inference
does not read those tensors.

Usage::

    python tools/strip_text_branch.py \\
        --input  output/artifacts/exp052_gap_generated_rationale_a00/fold_00/best.ckpt \\
        --output output/artifacts/exp052_gap_generated_rationale_a00/fold_00/best_stripped.ckpt \\
        --report docs/analysis/strip_text_branch_reports/exp052_a00_fold_00.md
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Key prefixes that are SAFE to drop (train-time-only).
STRIP_PREFIXES: tuple[str, ...] = (
    "model.text_head.",
    "model.part_head.",
    # Lightning wraps the model at ``model.*``; when a raw model is saved the
    # head lives at ``text_head.*`` / ``part_head.*`` as well.
    "text_head.",
    "part_head.",
)

# Exact keys that are SAFE to drop (train-time-only buffers).
STRIP_EXACT: frozenset[str] = frozenset({
    "scenario_embedding_matrix",
    "part_embedding_tensor",
    "part_mask_tensor",
})


def classify_key(key: str) -> bool:
    """Return True if the state_dict key should be stripped."""
    if key in STRIP_EXACT:
        return True
    for prefix in STRIP_PREFIXES:
        if key.startswith(prefix):
            return True
    return False


def strip_state_dict(state_dict: dict) -> tuple[dict, list[tuple[str, tuple[int, ...]]]]:
    """Return ``(kept_state_dict, removed_entries)``.

    ``removed_entries`` is a list of ``(key, shape)`` tuples for reporting.
    """
    kept: dict = {}
    removed: list[tuple[str, tuple[int, ...]]] = []
    for k, v in state_dict.items():
        if classify_key(k):
            shape = tuple(v.shape) if hasattr(v, "shape") else ()
            removed.append((k, shape))
        else:
            kept[k] = v
    return kept, removed


def write_report(
    report_path: Path,
    input_path: Path,
    output_path: Path,
    removed: list[tuple[str, tuple[int, ...]]],
    n_kept: int,
) -> None:
    lines = [
        f"# Strip-text-branch report",
        "",
        f"- input: `{input_path}`",
        f"- output: `{output_path}`",
        f"- generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        f"- kept state_dict keys: **{n_kept}**",
        f"- removed state_dict keys: **{len(removed)}**",
        "",
    ]
    if removed:
        lines.append("| key | shape |")
        lines.append("|---|---|")
        for k, s in removed:
            lines.append(f"| `{k}` | `{list(s)}` |")
    else:
        lines.append("_No train-time-only keys found; the checkpoint is already inference-clean._")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "output_path", required=True, type=click.Path(dir_okay=False))
@click.option("--report", "report_path", default=None, type=click.Path(dir_okay=False),
              help="Optional markdown report of removed keys.")
def main(input_path: str, output_path: str, report_path: str | None) -> None:
    ip = Path(input_path)
    op = Path(output_path)
    ckpt = torch.load(ip, map_location="cpu", weights_only=False)

    # Lightning stores the state_dict under "state_dict"; raw torch.save uses
    # either a bare dict or a wrapper dict under a caller-chosen key.
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
        kept, removed = strip_state_dict(sd)
        ckpt["state_dict"] = kept
        # Also scrub any Lightning hyperparameters that reference stripped tensors.
        hparams = ckpt.get("hyper_parameters") or ckpt.get("hparams") or {}
        if isinstance(hparams, dict):
            for dropped_key in ("scenario_embedding_matrix", "part_embedding_tensor",
                                "part_mask_tensor", "filename_to_scenario_idx",
                                "filename_to_part_idx"):
                hparams.pop(dropped_key, None)
        payload = ckpt
    elif isinstance(ckpt, dict):
        # Raw state_dict.
        kept, removed = strip_state_dict(ckpt)
        payload = kept
    else:
        raise click.ClickException(
            f"unsupported checkpoint structure in {ip}; expected dict or Lightning ckpt"
        )

    op.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, op)
    click.echo(
        f"Saved: {op} (kept={len(kept)}, removed={len(removed)})"
    )
    if removed:
        for k, s in removed[:5]:
            click.echo(f"  - dropped {k} shape={list(s)}")
        if len(removed) > 5:
            click.echo(f"  - ... and {len(removed) - 5} more")

    if report_path:
        write_report(
            Path(report_path), ip, op, removed, n_kept=len(kept),
        )
        click.echo(f"Report: {report_path}")


if __name__ == "__main__":
    main()
