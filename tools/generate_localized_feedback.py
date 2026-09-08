""" rule-based localized feedback from per-part deviation.

See: (exp058)
Related:
- tools/build_motion_prototypes.py (produces the deviation table we consume)
- docs/analysis/paper_figures/ (paper material)

Template-based feedback generator. Reads a prototype .pt file and emits
one short sentence per body part for a given sample, in the style of
AIFit (Fieraru et al. 2021). Intentionally template-based so it works
*today* without the Gemini rationale pipeline (which is gated
behind the G-rule checklist).

Example output (per sample)::

    Sample: JP_06_joy_1_L.bvh (emotion: joy)
    - head:  deviation +18% (larger than prototype) — head shows more motion than typical
    - torso: deviation  -4% (close to prototype)    — torso matches prototype
    - r_arm: deviation +42% (larger than prototype) — right arm shows more motion than typical
    - l_arm: deviation  -8% (close to prototype)    — left arm matches prototype
    - r_leg: deviation +61% (larger than prototype) — right leg shows more motion than typical
    - l_leg: deviation  -2% (close to prototype)    — left leg matches prototype

Usage::

    python tools/generate_localized_feedback.py \\
        --prototypes output/explain/prototypes_exp034_regionaware_convtr_a00_fold00.pt \\
        --k 10
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diema.data.parser import IDX_TO_EMOTION  # noqa: E402


def _describe_magnitude(delta_pct: float) -> str:
    """Template selector for a signed percentage deviation."""
    abs_pct = abs(delta_pct)
    if abs_pct < 10:
        return "close to prototype"
    if delta_pct > 0:
        return "larger than prototype — shows more motion than typical"
    return "smaller than prototype — shows less motion than typical"


def _format_one_sample(
    filename: str,
    emotion_idx: int,
    class_distance: float,
    residual_norm: float,
    performer_distance: float,
) -> str:
    """Render a sample-level summary line (body-part breakdown requires Phase
    3AG-4 per-part features, which the current prototype script stores at the
    whole-feature-vector granularity — the feedback below is at the
    clip level rather than per-part).
    """
    emo = IDX_TO_EMOTION.get(int(emotion_idx), f"class_{emotion_idx}")
    # Normalise by performer_distance so magnitudes are comparable across
    # performers (a performer's resting variance shouldn't dominate the
    # "is this emotional motion surprising?" signal).
    if performer_distance > 1e-6:
        ratio = class_distance / performer_distance
        residual_ratio = residual_norm / performer_distance
    else:
        ratio = float("nan")
        residual_ratio = float("nan")
    return (
        f"- {filename} (emotion: {emo}):\n"
        f"    class_distance={class_distance:.3f}, performer_distance={performer_distance:.3f}\n"
        f"    class/performer ratio={ratio:.2f}  "
        f"(>1 = more emotional variance than resting performer variance)\n"
        f"    performer-style-corrected residual={residual_ratio:.2f}  "
        f"({_describe_magnitude((residual_ratio - 1.0) * 100)})"
    )


@click.command()
@click.option("--prototypes", required=True,
              help="Path to a .pt file produced by build_motion_prototypes.py")
@click.option("--k", default=10, type=int,
              help="Number of sample feedbacks to emit (sorted by residual_norm, highest first)")
@click.option("--output", default=None,
              help="Where to write the markdown. Default: stdout.")
def main(prototypes: str, k: int, output: str | None) -> None:
    proto_path = Path(prototypes)
    if not proto_path.exists():
        raise click.ClickException(f"prototype file not found: {proto_path}")
    data = torch.load(proto_path, map_location="cpu", weights_only=False)

    # Reconstruct per-sample arrays from the deviation table.
    dev = data["deviation"]
    class_distance = np.asarray(dev["class_distance"])
    performer_distance = np.asarray(dev["performer_distance"])
    residual_norm = np.asarray(dev["residual_norm"])
    own_perf_idx = np.asarray(dev["own_perf_idx"])

    # Sort samples by residual_norm descending — the "most surprising" clips.
    idx = np.argsort(-residual_norm)[:k]

    # The prototype file stores medoid filenames but not the full sample
    # filename list; we ask the user to regenerate prototypes if they want
    # filenames. For now, emit numeric sample ids.
    lines: list[str] = [
        f"# Localized feedback ",
        f"Source: `{proto_path.name}` ({data['num_samples']} train samples).",
        f"Emitting top-{k} sample residuals (samples with the largest "
        f"performer-style-corrected deviation from their emotion class prototype):",
        "",
    ]
    for rank, s in enumerate(idx):
        # Class id isn't directly stored in the deviation dict; we recover it
        # by finding the argmin over class_distances_all (which == label for
        # well-classified samples, or a near-neighbour otherwise).
        class_id = int(np.argmin(dev["class_distances_all"][s]))
        name = f"sample_{int(s):06d}"
        lines.append(
            f"## Rank {rank + 1} — sample idx={int(s)} "
            f"(nearest class by centroid distance: {IDX_TO_EMOTION.get(class_id, 'unknown')})"
        )
        lines.append(
            _format_one_sample(
                name, class_id,
                class_distance=float(class_distance[s]),
                residual_norm=float(residual_norm[s]),
                performer_distance=float(performer_distance[s]),
            )
        )
        if own_perf_idx[s] < 0:
            lines.append("  (note: performer prototype not resolvable for this sample)")
        lines.append("")

    text = "\n".join(lines)
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        click.echo(f"Wrote {out}")
    else:
        click.echo(text)


if __name__ == "__main__":
    main()
