""" per-joint (single-node) masking faithfulness eval.

See: → body-part importance (per-joint saliency)
Related:
- tools/explain_part_masking.py (6-group counterpart — same checkpoint loader,
  same val loader, same F1/AUC math)
- diema/models/skeleton_graph.py::DIEMA_JOINT_NAMES (source of truth for 25 names)

Masks one CTV node at a time (25 joints) instead of the 6 anatomical
groups from, so the heatmap has one row per joint. Everything
else (orderings, AUCs) matches the part-masking tool so the two outputs
can be viewed side-by-side.

Usage::

    python tools/explain_joint_masking.py --experiment exp034_regionaware_convtr_a00 --fold all
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

from utils.env import EnvConfig  # noqa: E402
from diema.models import build_model  # noqa: E402
import diema.models  # noqa: F401, E402
from diema.models.skeleton_graph import DIEMA_JOINT_NAMES  # noqa: E402
from diema.data.parser import IDX_TO_EMOTION  # noqa: E402

from tools.infer_test_ensemble import _cfg_to_builder_ns, _load_checkpoint_into_model  # noqa: E402
from tools.explain_part_masking import (  # noqa: E402
    _build_val_loader,
    _forward_with_mask,
    _macro_f1,
    _per_class_f1,
    _auc_drop,
    NUM_CLASS,
)


def _faithfulness_curve(model, loader, device, order: list[int],
                        labels_baseline) -> list[float]:
    """Cumulatively mask joints in ``order`` (each element is a CTV node id).

    Returns Macro-F1 at k=0..len(order). Reuses `_forward_with_mask` which
    accepts a tuple of node ids as its mask.
    """
    curve: list[float] = []
    for k in range(len(order) + 1):
        mask_nodes = tuple(order[:k]) if k > 0 else None
        logits, labels = _forward_with_mask(model, loader, device, mask_nodes)
        assert np.array_equal(labels, labels_baseline), (
            "val label ordering changed between mask steps"
        )
        curve.append(_macro_f1(labels, logits.argmax(1)))
    return curve


@click.command()
@click.option("--experiment", required=True)
@click.option("--fold", default="0", help="Fold number or 'all'")
@click.option("--num-folds", default=10, type=int)
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]))
@click.option("--batch-size", default=128, type=int)
@click.option("--top-k-faithfulness", default=10, type=int,
              help="Number of top/bottom joints to cumulatively mask in the curves (default 10)")
@click.option("--output-dir", default=None)
def main(experiment: str, fold: str, num_folds: int, device: str,
         batch_size: int, top_k_faithfulness: int,
         output_dir: str | None) -> None:
    env = EnvConfig()
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    click.echo(f"Device: {dev}")

    exp_dir = env.artifacts_dir / experiment
    if fold == "all":
        fold_ids = [int(p.name.split("_")[1]) for p in sorted(exp_dir.glob("fold_[0-9][0-9]"))]
    else:
        fold_ids = [int(fold)]

    out_root = Path(output_dir) if output_dir else (env.project_root / "docs/analysis/joint_masking")
    out_root.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    n_joints = len(DIEMA_JOINT_NAMES)  # 25

    for f_id in fold_ids:
        fold_dir = exp_dir / f"fold_{f_id:02d}"
        ckpt = fold_dir / "best.ckpt"
        cfg_path = fold_dir / "config.yaml"
        if not (ckpt.exists() and cfg_path.exists()):
            click.echo(f"[skip] fold_{f_id:02d}: missing ckpt or config", err=True)
            continue

        cfg = OmegaConf.to_container(OmegaConf.load(str(cfg_path)), resolve=True)
        model = build_model(_cfg_to_builder_ns(cfg))
        _load_checkpoint_into_model(ckpt, model)
        loader = _build_val_loader(cfg, fold=f_id, num_folds=num_folds, batch_size=batch_size)

        # 1. Baseline
        base_logits, labels = _forward_with_mask(model, loader, dev, None)
        if labels.size == 0:
            click.echo(f"[skip] fold_{f_id:02d}: empty val", err=True)
            continue
        base_pred = base_logits.argmax(1)
        base_f1 = _macro_f1(labels, base_pred)
        base_per_class = _per_class_f1(labels, base_pred)
        click.echo(f"\nfold_{f_id:02d} baseline Macro-F1: {base_f1:.4f}")

        # 2. Per-joint single mask
        per_joint_f1: dict[int, float] = {}
        per_joint_per_class: dict[int, np.ndarray] = {}
        for j in range(n_joints):
            logits_j, _ = _forward_with_mask(model, loader, dev, (j,))
            pred_j = logits_j.argmax(1)
            per_joint_f1[j] = _macro_f1(labels, pred_j)
            per_joint_per_class[j] = _per_class_f1(labels, pred_j)
        # Concise summary (top-3 / bottom-3)
        ranked = sorted(range(n_joints), key=lambda j: -(base_f1 - per_joint_f1[j]))
        click.echo("  top-3 most-important joints (largest F1 drop):")
        for j in ranked[:3]:
            click.echo(
                f"    [{j:02d}] {DIEMA_JOINT_NAMES[j]:>14} "
                f"F1 {per_joint_f1[j]:.4f}  drop {base_f1 - per_joint_f1[j]:+.4f}"
            )
        click.echo("  bottom-3 least-important joints:")
        for j in ranked[-3:]:
            click.echo(
                f"    [{j:02d}] {DIEMA_JOINT_NAMES[j]:>14} "
                f"F1 {per_joint_f1[j]:.4f}  drop {base_f1 - per_joint_f1[j]:+.4f}"
            )

        # 3. Faithfulness curve (limited to top_k for cost)
        drops_by_joint = {j: base_f1 - per_joint_f1[j] for j in range(n_joints)}
        top_order = sorted(range(n_joints), key=lambda j: -drops_by_joint[j])[:top_k_faithfulness]
        reverse_order = sorted(range(n_joints), key=lambda j: drops_by_joint[j])[:top_k_faithfulness]
        rng = np.random.default_rng(f_id)
        random_order = list(rng.choice(n_joints, size=top_k_faithfulness, replace=False))

        curves = {
            "important_first": _faithfulness_curve(model, loader, dev, top_order, labels),
            "reverse": _faithfulness_curve(model, loader, dev, reverse_order, labels),
            "random": _faithfulness_curve(model, loader, dev, random_order, labels),
        }
        aucs = {k: _auc_drop(c) for k, c in curves.items()}
        click.echo(
            f"  faithfulness AUC (top-{top_k_faithfulness} cumulative) — "
            f"important_first: {aucs['important_first']:.4f}, "
            f"reverse: {aucs['reverse']:.4f}, "
            f"random: {aucs['random']:.4f}"
        )

        # Record
        for j in range(n_joints):
            row = {
                "experiment": experiment,
                "fold": f_id,
                "joint_idx": j,
                "joint_name": DIEMA_JOINT_NAMES[j],
                "baseline_f1": base_f1,
                "masked_f1": per_joint_f1[j],
                "f1_drop": base_f1 - per_joint_f1[j],
            }
            for c in range(NUM_CLASS):
                row[f"f1_drop_{IDX_TO_EMOTION[c]}"] = float(
                    base_per_class[c] - per_joint_per_class[j][c]
                )
            all_records.append(row)

        # Per-fold JSON with the curves
        (out_root / f"{experiment}_fold{f_id:02d}_joint_faithfulness.json").write_text(
            json.dumps({
                "experiment": experiment,
                "fold": f_id,
                "baseline_f1": base_f1,
                "top_k": top_k_faithfulness,
                "top_order_joint_idx": top_order,
                "reverse_order_joint_idx": reverse_order,
                "random_order_joint_idx": list(map(int, random_order)),
                "curves": curves,
                "aucs": aucs,
            }, indent=2)
        )

    # Aggregate CSV
    df = pd.DataFrame(all_records)
    csv_path = out_root / f"{experiment}_joint_importance.csv"
    df.to_csv(csv_path, index=False)
    click.echo(f"\nSaved: {csv_path}  ({len(df)} rows)")

    # Heatmap: joints × emotions
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        agg = df.groupby("joint_idx").mean(numeric_only=True)
        drop_cols = [c for c in agg.columns if c.startswith("f1_drop_")]
        heatmap = agg[drop_cols].reindex(range(n_joints))
        heatmap.columns = [c.replace("f1_drop_", "") for c in heatmap.columns]
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(heatmap.values, aspect="auto", cmap="RdYlBu_r",
                       vmin=-0.03, vmax=0.15)
        ax.set_xticks(range(len(heatmap.columns)))
        ax.set_xticklabels(heatmap.columns, rotation=45, ha="right")
        ax.set_yticks(range(n_joints))
        ax.set_yticklabels([
            f"{j:02d} {DIEMA_JOINT_NAMES[j]}" for j in range(n_joints)
        ])
        ax.set_xlabel("Emotion class")
        ax.set_ylabel("CTV joint (single-node zero-mask)")
        fig.colorbar(im, ax=ax, label="Macro-F1 drop")
        fig.tight_layout()
        fig_path = out_root / f"{experiment}_joint_heatmap.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        click.echo(f"Saved heatmap: {fig_path}")
    except Exception as exc:
        click.echo(f"[warn] heatmap skipped: {exc}", err=True)


if __name__ == "__main__":
    main()
