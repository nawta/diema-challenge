""" temporal-window masking faithfulness eval.

See: (exp058)
Related:
- tools/explain_part_masking.py (space-axis counterpart)
- diema/training/callbacks.py::OOFPredictionCallback

For a trained experiment, this tool:

1. Rebuilds the model from ``config.yaml`` and loads best.ckpt.
2. Computes a **gradient-based** per-timestep saliency for every val sample:
   ``s[t] = || d(logit_true_class) / d(x[:, t, :]) ||_2``. This uses the
   ground-truth class label (not the prediction) so that masking is evaluated
   against the supervision signal.
3. Splits the ``T`` timesteps into ``num_windows`` equal chunks, ranks them
   by mean saliency, and cumulatively zeros the top-k windows.
4. Reports Macro-F1 at each k for three orderings:
   - ``salient_first``: highest-saliency window first
   - ``reverse``: least salient first
   - ``random``: seeded null baseline
5. Outputs per-fold JSON + aggregated CSV + (optional) saliency heatmap PNG.

Temporal saliency source: **gradient** (no rationale cache / attention API
dependency). Gemini rationale source can be swapped in once
 Go gate is cleared.

Usage::

    python tools/explain_temporal_masking.py \\
        --experiment exp034_regionaware_convtr_a00 \\
        --fold all \\
        --num-windows 4 \\
        --device cuda
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
from diema.data.parser import IDX_TO_EMOTION  # noqa: E402

from tools.infer_test_ensemble import _cfg_to_builder_ns, _load_checkpoint_into_model  # noqa: E402
from tools.explain_part_masking import _macro_f1, _auc_drop, _build_val_loader  # noqa: E402


NUM_CLASS = 12


def _per_sample_temporal_saliency(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Return ``(N, T)`` saliency computed as the L2 norm of d(logit_y)/dx per frame.

    Uses a fresh gradient computation each call so the model weights are left
    untouched. Model is expected to be on ``x.device`` and in eval() mode.
    """
    x_grad = x.detach().clone().requires_grad_(True)
    out = model(x_grad)
    logits = out["logits"] if isinstance(out, dict) else out
    # Select the logit of the true class for each row.
    sel = logits.gather(1, y.view(-1, 1)).squeeze(1)
    # Sum over the batch to get a single scalar to backward; because each sample's
    # input is independent in the graph, the resulting per-input gradient equals
    # the per-sample gradient of its own selected logit. This is NOT vmap-style
    # per-sample-grad (which avoids cross-sample graph sharing); it works here
    # because the model treats each sample independently on the batch axis.
    sel.sum().backward()
    grad = x_grad.grad  # (N, C, T, V)
    if grad is None:
        return torch.zeros(x.size(0), x.size(2), device=x.device)
    # L2 over (C, V) → (N, T)
    sal = grad.abs().pow(2).sum(dim=(1, 3)).sqrt()
    return sal.detach()


def _compute_saliency_over_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate per-sample ``(T,)`` saliency tensors + labels over the loader."""
    model = model.to(device).eval()
    # Turn off param grads so backward() only populates x_grad.
    for p in model.parameters():
        p.requires_grad_(False)

    sals: list[torch.Tensor] = []
    lbls: list[torch.Tensor] = []
    for batch in loader:
        x, y, _fn = batch
        x = x.to(device)
        y = y.to(device)
        sal = _per_sample_temporal_saliency(model, x, y)
        sals.append(sal.cpu())
        lbls.append(y.cpu())
    return torch.cat(sals, 0), torch.cat(lbls, 0)


@torch.no_grad()
def _infer_with_temporal_mask(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    per_sample_mask: torch.Tensor | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward with a per-sample (N, T) bool mask. ``True`` → zero that frame.

    Used for both the unmasked baseline (mask=None) and all ranked-masking
    experiments. ``per_sample_mask`` must match the loader's sample order.
    """
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    cursor = 0
    for batch in loader:
        x, y, _fn = batch
        bs = x.size(0)
        x = x.to(device)
        if per_sample_mask is not None:
            mask_chunk = per_sample_mask[cursor:cursor + bs].to(device)  # (bs, T)
            # Broadcast to (bs, 1, T, 1) and multiply-zero.
            x = x.clone()
            x[mask_chunk.unsqueeze(1).unsqueeze(-1).expand_as(x).bool()] = 0.0
        cursor += bs
        out = model(x)
        logits = out["logits"] if isinstance(out, dict) else out
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_logits, 0), np.concatenate(all_labels, 0)


def _window_ranking(
    saliency: torch.Tensor,
    num_windows: int,
) -> torch.Tensor:
    """Split each sample's (T,) saliency into ``num_windows`` chunks,
    return ``(N, num_windows)`` int tensor with the ranking index
    (0 = most salient, num_windows-1 = least)."""
    N, T = saliency.shape
    # If T is not divisible, the last window absorbs the remainder.
    base = T // num_windows
    splits: list[torch.Tensor] = []
    for w in range(num_windows):
        start = w * base
        end = (w + 1) * base if w < num_windows - 1 else T
        # Mean saliency in the window.
        splits.append(saliency[:, start:end].mean(dim=1))
    means = torch.stack(splits, dim=1)  # (N, num_windows)
    # Rank descending (largest mean saliency → rank 0).
    ranks = means.argsort(dim=1, descending=True)  # (N, num_windows) index of window
    return ranks


def _mask_for_top_k(
    ranks: torch.Tensor,
    num_windows: int,
    T: int,
    top_k: int,
    order: str,
    seed: int,
) -> torch.Tensor:
    """Build a per-sample ``(N, T)`` bool mask for the chosen top-k windows.

    ``order``:
      - ``"salient_first"``: pick the k highest-saliency windows (rank 0..k-1)
      - ``"reverse"``: pick the k lowest-saliency windows (rank -k..-1)
      - ``"random"``: deterministic random subset of k windows, seeded by
        ``seed + sample_index``.
    """
    N = ranks.size(0)
    base = T // num_windows
    mask = torch.zeros(N, T, dtype=torch.bool)
    if top_k == 0:
        return mask

    if order == "salient_first":
        chosen = ranks[:, :top_k]
    elif order == "reverse":
        chosen = ranks[:, -top_k:]
    elif order == "random":
        rng = np.random.default_rng(seed)
        chosen_np = np.stack(
            [rng.choice(num_windows, size=top_k, replace=False) for _ in range(N)],
            axis=0,
        )
        chosen = torch.from_numpy(chosen_np).long()
    else:
        raise ValueError(f"unknown order {order!r}")

    for i in range(N):
        for w in chosen[i].tolist():
            start = w * base
            end = (w + 1) * base if w < num_windows - 1 else T
            mask[i, start:end] = True
    return mask


def _faithfulness_curve(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    saliency: torch.Tensor,
    num_windows: int,
    order: str,
    fold_seed: int,
    T: int,
) -> list[float]:
    """Run F1 evaluation at k=0..num_windows for a given ordering."""
    ranks = _window_ranking(saliency, num_windows)
    curve: list[float] = []
    for k in range(num_windows + 1):
        mask = _mask_for_top_k(ranks, num_windows, T, k, order, seed=fold_seed)
        if k == 0:
            logits, labels = _infer_with_temporal_mask(model, loader, device, None)
        else:
            logits, labels = _infer_with_temporal_mask(model, loader, device, mask)
        curve.append(_macro_f1(labels, logits.argmax(1)))
    return curve


@click.command()
@click.option("--experiment", required=True)
@click.option("--fold", default="0", help="Fold number or 'all'")
@click.option("--num-folds", default=10, type=int)
@click.option("--num-windows", default=4, type=int,
              help="Split T frames into this many equal chunks for masking")
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]))
@click.option("--batch-size", default=128, type=int)
@click.option("--output-dir", default=None)
def main(experiment: str, fold: str, num_folds: int, num_windows: int,
         device: str, batch_size: int, output_dir: str | None) -> None:
    env = EnvConfig()
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    click.echo(f"Device: {dev}")

    exp_dir = env.artifacts_dir / experiment
    if fold == "all":
        fold_ids = [int(p.name.split("_")[1]) for p in sorted(exp_dir.glob("fold_[0-9][0-9]"))]
    else:
        fold_ids = [int(fold)]

    out_root = Path(output_dir) if output_dir else (env.project_root / "docs/analysis/temporal_masking")
    out_root.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for f_id in fold_ids:
        fold_dir = exp_dir / f"fold_{f_id:02d}"
        ckpt = fold_dir / "best.ckpt"
        cfg_path = fold_dir / "config.yaml"
        if not (ckpt.exists() and cfg_path.exists()):
            click.echo(f"[skip] fold_{f_id:02d}: missing ckpt or config", err=True)
            continue

        cfg_dict = OmegaConf.to_container(OmegaConf.load(str(cfg_path)), resolve=True)
        builder_ns = _cfg_to_builder_ns(cfg_dict)
        model = build_model(builder_ns)
        _load_checkpoint_into_model(ckpt, model)

        loader = _build_val_loader(cfg_dict, fold=f_id, num_folds=num_folds,
                                   batch_size=batch_size)
        T = cfg_dict.get("clip_length", 64)

        click.echo(f"\nfold_{f_id:02d}: computing gradient saliency...")
        saliency, _labels = _compute_saliency_over_loader(model, loader, dev)

        # Baseline F1
        logits0, labels0 = _infer_with_temporal_mask(model, loader, dev, None)
        base_f1 = _macro_f1(labels0, logits0.argmax(1))
        click.echo(f"  baseline Macro-F1: {base_f1:.4f}")

        curves = {}
        for order in ("salient_first", "reverse", "random"):
            curves[order] = _faithfulness_curve(
                model, loader, dev, saliency, num_windows, order,
                fold_seed=f_id, T=T,
            )
        aucs = {o: _auc_drop(c) for o, c in curves.items()}
        click.echo(
            f"  AUC — salient_first: {aucs['salient_first']:.4f}, "
            f"reverse: {aucs['reverse']:.4f}, "
            f"random: {aucs['random']:.4f}"
        )

        # Save per-fold JSON
        payload = {
            "experiment": experiment,
            "fold": f_id,
            "num_windows": num_windows,
            "T": T,
            "baseline_f1": base_f1,
            "curves": curves,
            "aucs": aucs,
            "saliency_source": "gradient",
        }
        out_json = out_root / f"{experiment}_fold{f_id:02d}_temporal_faithfulness.json"
        out_json.write_text(json.dumps(payload, indent=2))

        all_records.append({
            "experiment": experiment,
            "fold": f_id,
            "baseline_f1": base_f1,
            "num_windows": num_windows,
            **{f"auc_{o}": a for o, a in aucs.items()},
        })

    df = pd.DataFrame(all_records)
    csv_path = out_root / f"{experiment}_temporal_importance.csv"
    df.to_csv(csv_path, index=False)
    click.echo(f"\nSaved: {csv_path}  ({len(df)} rows)")

    # Optional: aggregated 2D heatmap of saliency
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Plot the mean AUC per ordering across folds as a bar chart.
        if len(df) > 0:
            means = {o: df[f"auc_{o}"].mean() for o in ("salient_first", "reverse", "random")}
            stds = {o: df[f"auc_{o}"].std() for o in ("salient_first", "reverse", "random")}
            fig, ax = plt.subplots(figsize=(6, 4))
            xs = list(means.keys())
            ys = [means[o] for o in xs]
            es = [stds[o] for o in xs]
            ax.bar(xs, ys, yerr=es, capsize=4)
            ax.set_ylabel("Faithfulness AUC")
            ax.set_title(f"Temporal masking AUC — {experiment}")
            fig.tight_layout()
            fig_path = out_root / f"{experiment}_temporal_auc_bar.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            click.echo(f"Saved bar chart: {fig_path}")
    except Exception as exc:
        click.echo(f"[warn] figure skipped: {exc}", err=True)


if __name__ == "__main__":
    main()
