""" counterfactual motion edits on body-part channels.

See: (exp058)
Related:
- tools/explain_part_masking.py (shares checkpoint + val-loader helpers)
- diema/models/skeleton_graph.py::DIEMA_BODY_PARTS

For a trained experiment, this tool:

1. Rebuilds the model from ``config.yaml`` and loads best.ckpt.
2. For each val sample, applies a programmatic counterfactual edit to
   one body part's temporal trajectory. Three edit types, each applied
   independently per body part:
   - ``freeze``: replace the part's per-frame values by their temporal
     mean ⇒ pose preserved, motion zeroed
   - ``damp`` (default factor 0.5): centre-scale the part's temporal
     variation by 0.5 ⇒ smaller motion around the mean pose
   - ``amplify`` (default factor 2.0): centre-scale by 2.0 ⇒ larger
     motion around the mean pose
3. Forwards original and edited inputs, records:
   - argmax prediction shift
   - softmax probability of the ground-truth class (before / after / Δ)
   - top-K logit change magnitude
4. Writes per-sample CSV and a 4×4 paper figure showing average Δprob for
   each (part × edit) cell — so the reader sees e.g. "freezing the head
   consistently pulls pride → gratitude".

Why centre-scale (not multiply raw channels): the raw channels encode
absolute joint rotation / position; scaling them destroys the pose. We
want to keep the pose and manipulate the *variation*, which is what
centre-scaling does (subtract mean, scale, add mean back).

Usage::

    python tools/counterfactual_motion_edit.py \\
        --experiment exp034_regionaware_convtr_a00 --fold 0 --device cuda
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pybvh_ml  # noqa: E402
from utils.env import EnvConfig  # noqa: E402
from diema.models import build_model  # noqa: E402
import diema.models  # noqa: F401, E402
from diema.models.skeleton_graph import DIEMA_BODY_PARTS  # noqa: E402
from diema.data.parser import IDX_TO_EMOTION  # noqa: E402

from tools.infer_test_ensemble import _cfg_to_builder_ns, _load_checkpoint_into_model  # noqa: E402
from tools.explain_part_masking import _build_val_loader  # noqa: E402


PARTS_ORDERED = list(DIEMA_BODY_PARTS.keys())
EDIT_TYPES = ("freeze", "damp_0.5", "amplify_2.0")
NUM_CLASS = 12


def _apply_edit(x: torch.Tensor, nodes: tuple[int, ...], edit: str) -> torch.Tensor:
    """Return a NEW tensor with the edit applied to ``x[:, :, :, nodes]``.

    ``x`` shape: (N, C, T, V). Edit types:
    - ``freeze``       : x[n,c,t,v] ← mean_t(x[n,c,:,v])
    - ``damp_0.5``     : x[n,c,t,v] ← mean + 0.5 * (x[n,c,t,v] - mean)
    - ``amplify_2.0``  : x[n,c,t,v] ← mean + 2.0 * (x[n,c,t,v] - mean)

    The mean is computed over the temporal axis per (sample, channel, node)
    so the resulting pose (= long-term average channel value) is preserved.
    """
    out = x.clone()
    nodes_arr = list(nodes)
    slc = out[:, :, :, nodes_arr]  # (N, C, T, |nodes|)
    mean = slc.mean(dim=2, keepdim=True)  # (N, C, 1, |nodes|)
    if edit == "freeze":
        out[:, :, :, nodes_arr] = mean.expand_as(slc)
    elif edit.startswith("damp_") or edit.startswith("amplify_"):
        factor = float(edit.split("_", maxsplit=1)[1])
        out[:, :, :, nodes_arr] = mean + factor * (slc - mean)
    else:
        raise ValueError(f"unknown edit {edit!r}")
    return out


@torch.no_grad()
def _run_edits(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sweep every (body_part, edit) combo, return a per-sample frame + mean-shift matrix."""
    model = model.to(device).eval()
    records: list[dict] = []
    # Accumulator for per-(part, edit) mean Δprob and Δargmax-equal
    # indexed (part_idx, edit_idx, class) using running sum.
    delta_prob_sum = np.zeros(
        (len(PARTS_ORDERED), len(EDIT_TYPES), NUM_CLASS), dtype=np.float64
    )
    counts = np.zeros(
        (len(PARTS_ORDERED), len(EDIT_TYPES), NUM_CLASS), dtype=np.int64
    )

    sample_offset = 0
    for batch in loader:
        x, y, fn = batch
        x = x.to(device)
        y = y.to(device)
        # Baseline forward
        out_base = model(x)
        logits_base = out_base["logits"] if isinstance(out_base, dict) else out_base
        probs_base = torch.softmax(logits_base, dim=1)
        pred_base = logits_base.argmax(dim=1)

        for pi, part in enumerate(PARTS_ORDERED):
            nodes = tuple(DIEMA_BODY_PARTS[part])
            for ei, edit in enumerate(EDIT_TYPES):
                x_edit = _apply_edit(x, nodes, edit)
                out_edit = model(x_edit)
                logits_edit = out_edit["logits"] if isinstance(out_edit, dict) else out_edit
                probs_edit = torch.softmax(logits_edit, dim=1)
                pred_edit = logits_edit.argmax(dim=1)

                # per-sample stats
                for b in range(x.size(0)):
                    idx = sample_offset + b
                    y_true = int(y[b].item())
                    delta = float(probs_edit[b, y_true].item() - probs_base[b, y_true].item())
                    records.append({
                        "sample_idx": idx,
                        "filename": fn[b],
                        "true_class": IDX_TO_EMOTION[y_true],
                        "part": part,
                        "edit": edit,
                        "p_true_before": float(probs_base[b, y_true].item()),
                        "p_true_after": float(probs_edit[b, y_true].item()),
                        "delta_p_true": delta,
                        "pred_before": IDX_TO_EMOTION[int(pred_base[b].item())],
                        "pred_after": IDX_TO_EMOTION[int(pred_edit[b].item())],
                        "flipped": bool(pred_base[b].item() != pred_edit[b].item()),
                    })
                    delta_prob_sum[pi, ei, y_true] += delta
                    counts[pi, ei, y_true] += 1

        sample_offset += x.size(0)

    # Average Δ per (part, edit, class) over samples belonging to that class
    mean_delta = np.zeros_like(delta_prob_sum)
    mask = counts > 0
    mean_delta[mask] = delta_prob_sum[mask] / counts[mask]
    return pd.DataFrame(records), mean_delta


def _figure_heatmap(mean_delta: np.ndarray, out_base: Path, experiment: str) -> None:
    """Render a (parts × edits) × emotions heatmap of mean Δp_true."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    # Stack parts × edits on rows, emotions on columns.
    n_parts, n_edits, n_class = mean_delta.shape
    stacked = mean_delta.reshape(n_parts * n_edits, n_class)
    row_labels = [f"{p}  {e}" for p in PARTS_ORDERED for e in EDIT_TYPES]
    col_labels = [IDX_TO_EMOTION[c] for c in range(n_class)]

    vmax = max(abs(stacked.min()), abs(stacked.max()), 0.05)
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(stacked, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_class))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(stacked.shape[0]))
    ax.set_yticklabels(row_labels)
    ax.set_title(f"Counterfactual edit effect — mean Δ p(true class)  [{experiment}]")
    fig.colorbar(im, ax=ax, label="Δ prob (edited − baseline)")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"))
    plt.close(fig)


@click.command()
@click.option("--experiment", required=True)
@click.option("--fold", default=0, type=int)
@click.option("--num-folds", default=10, type=int)
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]))
@click.option("--batch-size", default=128, type=int)
@click.option("--output-dir", default=None)
def main(experiment: str, fold: int, num_folds: int, device: str,
         batch_size: int, output_dir: str | None) -> None:
    env = EnvConfig()
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    click.echo(f"Device: {dev}")

    fold_dir = env.artifacts_dir / experiment / f"fold_{fold:02d}"
    ckpt = fold_dir / "best.ckpt"
    cfg_path = fold_dir / "config.yaml"
    if not (ckpt.exists() and cfg_path.exists()):
        raise click.ClickException(f"missing best.ckpt or config.yaml in {fold_dir}")

    cfg = OmegaConf.to_container(OmegaConf.load(str(cfg_path)), resolve=True)
    model = build_model(_cfg_to_builder_ns(cfg))
    _load_checkpoint_into_model(ckpt, model)
    loader = _build_val_loader(cfg, fold=fold, num_folds=num_folds, batch_size=batch_size)

    out_root = (
        Path(output_dir) if output_dir
        else env.project_root / "docs/analysis/counterfactual"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    click.echo(f"Running counterfactual edits on fold_{fold:02d}...")
    df, mean_delta = _run_edits(model, loader, dev)
    click.echo(f"  {len(df)} per-sample records over {len(PARTS_ORDERED)} parts × "
               f"{len(EDIT_TYPES)} edits")

    csv_path = out_root / f"{experiment}_fold{fold:02d}_counterfactual.csv"
    df.to_csv(csv_path, index=False)
    click.echo(f"Saved: {csv_path}")

    fig_path = out_root / f"{experiment}_fold{fold:02d}_counterfactual_heatmap"
    _figure_heatmap(mean_delta, fig_path, experiment)
    click.echo(f"Saved figure: {fig_path}.pdf / .png")

    # Summary: top-K (part, edit) pairs by absolute mean Δp over all classes.
    summary = (
        df.groupby(["part", "edit"])["delta_p_true"]
        .mean().reset_index()
        .sort_values("delta_p_true")
    )
    summary_path = out_root / f"{experiment}_fold{fold:02d}_counterfactual_summary.md"
    lines = [
        f"# Counterfactual motion edit — {experiment} fold_{fold:02d}",
        "",
        "## Mean Δ p(true class) across ALL val samples",
        "",
        "Ranked from largest *drop* in the true-class probability (most harmful)"
        " to largest *gain*:",
        "",
        "| rank | part | edit | mean Δ p_true |",
        "|------|------|------|--------------|",
    ]
    for i, r in enumerate(summary.itertuples(), 1):
        lines.append(f"| {i} | {r.part} | {r.edit} | {r.delta_p_true:+.4f} |")
    lines.append("")
    lines.append(
        "Samples flipped by each edit:"
    )
    flips = df.groupby(["part", "edit"])["flipped"].mean().reset_index()
    flips["flipped_pct"] = (flips["flipped"] * 100).round(1)
    lines.append("")
    lines.append("| part | edit | flipped (%) |")
    lines.append("|------|------|-------------|")
    for r in flips.sort_values("flipped_pct", ascending=False).itertuples():
        lines.append(f"| {r.part} | {r.edit} | {r.flipped_pct} |")
    summary_path.write_text("\n".join(lines))
    click.echo(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
