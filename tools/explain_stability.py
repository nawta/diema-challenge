""" perturbation stability of part-importance ranking.

See: (exp058)
Related:
- tools/explain_part_masking.py (per-part F1 drop ranking we re-evaluate)

Tests whether the body-part importance ordering produced by
``explain_part_masking.py`` is **stable** under small input-space
perturbations. Method:

1. Load checkpoint + val fold.
2. For each σ in ``--sigmas`` (default: ``0.0, 0.005, 0.01, 0.02``),
   - Add isotropic Gaussian noise ``N(0, σ)`` to the val tensor in 6D /
     joint_pos space (whatever the training stream uses).
   - For each body-part, zero-mask and compute Macro-F1 drop ⇒ rank the
     6 parts by drop.
3. Compute pairwise **Spearman rank correlation** between the σ=0 ranking
   and every other σ's ranking.
4. Aggregate across folds. A robust explanation has correlation ≥ ~0.8
   at small σ; collapsing to ~0 means the explanation swings with noise.

The perturbation is applied AFTER the input tensor leaves the dataset
(so dataset-level normalization / augmentation is untouched). Noise scale
is in the same units as the training stream — for rotation_6d that is
dimensionless (~0.01 is a small perturbation; ~0.05 noticeable).

Usage::

    python tools/explain_stability.py \\
        --experiment exp034_regionaware_convtr_a00 \\
        --fold all \\
        --sigmas 0.0,0.005,0.01,0.02 \\
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
from diema.models.skeleton_graph import DIEMA_BODY_PARTS  # noqa: E402

from tools.infer_test_ensemble import _cfg_to_builder_ns, _load_checkpoint_into_model  # noqa: E402
from tools.explain_part_masking import _macro_f1, _build_val_loader  # noqa: E402


PARTS_ORDERED = list(DIEMA_BODY_PARTS.keys())


@torch.no_grad()
def _infer_with_noise_and_mask(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    noise_tensors: list[torch.Tensor] | None,
    mask_nodes: tuple[int, ...] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward with pre-sampled batch-aligned noise + optional part zero-mask.

    ``noise_tensors`` is a list whose i-th entry is the noise to add to the
    i-th mini-batch (or ``None`` for the σ=0 reference run). Pre-sampling the
    noise outside this loop is what makes the baseline and each mask share
    the same noise realisation, so the Spearman comparison at a given σ
    measures mask effect rather than noise variance (a critical review finding).
    """
    model = model.to(device).eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for i, batch in enumerate(loader):
        x, y, _fn = batch
        x = x.to(device)
        if noise_tensors is not None:
            # Broadcast-safe add — noise tensor shape already matches x.
            x = x + noise_tensors[i]
        if mask_nodes is not None:
            x = x.clone()
            for n in mask_nodes:
                x[:, :, :, n] = 0.0
        out = model(x)
        logits = out["logits"] if isinstance(out, dict) else out
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_logits, 0), np.concatenate(all_labels, 0)


def _sample_batch_noise(
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    noise_sigma: float,
    seed: int,
) -> list[torch.Tensor] | None:
    """Pre-sample a per-batch noise tensor list using a local Generator.

    Uses an isolated ``torch.Generator(device=...)`` seeded from ``seed`` so
    the global torch RNG (shared with DataLoader workers, augmentation, etc.)
    is untouched. Returns ``None`` when σ == 0 to signal the caller to skip
    the add.
    """
    if noise_sigma <= 0.0:
        return None
    gen = torch.Generator(device=device).manual_seed(int(seed))
    tensors: list[torch.Tensor] = []
    for batch in loader:
        x, _y, _fn = batch
        shape = (x.size(0),) + tuple(x.shape[1:])
        tensors.append(
            torch.randn(shape, generator=gen, device=device) * noise_sigma
        )
    return tensors


def _part_ranking_under_noise(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    noise_sigma: float,
    seed: int,
) -> tuple[dict[str, float], float]:
    """Return ``(drops_by_part, baseline_f1)`` at the given noise level.

    Generates one batch-aligned noise tensor list up front, then reuses it
    for the unmasked baseline and every part-mask forward, so the ranking is
    measured against a *single fixed* noise realisation rather than
    independent draws (a critical review finding).
    """
    noise_tensors = _sample_batch_noise(loader, device, noise_sigma, seed)
    base_logits, labels = _infer_with_noise_and_mask(
        model, loader, device, noise_tensors, None,
    )
    base_f1 = _macro_f1(labels, base_logits.argmax(1))
    drops: dict[str, float] = {}
    for p in PARTS_ORDERED:
        mask = tuple(DIEMA_BODY_PARTS[p])
        masked_logits, _ = _infer_with_noise_and_mask(
            model, loader, device, noise_tensors, mask,
        )
        drops[p] = base_f1 - _macro_f1(labels, masked_logits.argmax(1))
    return drops, base_f1


def _rank_order(drops: dict[str, float]) -> list[int]:
    """Return the argsort ranks (higher drop = smaller rank index = more important)."""
    # Map part → integer rank (0 = most important, 5 = least).
    sorted_parts = sorted(PARTS_ORDERED, key=lambda p: -drops[p])
    part_to_rank = {p: i for i, p in enumerate(sorted_parts)}
    return [part_to_rank[p] for p in PARTS_ORDERED]


def _spearman(a: list[int] | np.ndarray, b: list[int] | np.ndarray) -> float:
    """Spearman rank correlation between two rank vectors."""
    from scipy.stats import spearmanr
    rho, _ = spearmanr(a, b)
    return float(rho)


@click.command()
@click.option("--experiment", required=True)
@click.option("--fold", default="0", help="Fold number or 'all'")
@click.option("--num-folds", default=10, type=int)
@click.option("--sigmas", default="0.0,0.005,0.01,0.02",
              help="Comma-separated noise sigmas (default 0/0.5%/1%/2%)")
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]))
@click.option("--batch-size", default=128, type=int)
@click.option("--seed", default=42, type=int, help="RNG seed (per fold = seed + fold_id)")
@click.option("--output-dir", default=None)
def main(experiment: str, fold: str, num_folds: int, sigmas: str,
         device: str, batch_size: int, seed: int,
         output_dir: str | None) -> None:
    env = EnvConfig()
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    click.echo(f"Device: {dev}")

    sigma_list = [float(s.strip()) for s in sigmas.split(",")]
    if sigma_list[0] != 0.0:
        raise click.BadParameter("first sigma must be 0.0 (reference)")

    exp_dir = env.artifacts_dir / experiment
    if fold == "all":
        fold_ids = [int(p.name.split("_")[1]) for p in sorted(exp_dir.glob("fold_[0-9][0-9]"))]
    else:
        fold_ids = [int(fold)]

    out_root = Path(output_dir) if output_dir else (env.project_root / "docs/analysis/stability")
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

        click.echo(f"\nfold_{f_id:02d}")
        ref_drops, ref_base = _part_ranking_under_noise(
            model, loader, dev, noise_sigma=0.0, seed=seed + f_id
        )
        ref_rank = _rank_order(ref_drops)
        click.echo(f"  σ=0.000 baseline_f1={ref_base:.4f} ranking={ref_rank}")

        per_sigma: dict[float, dict] = {}
        per_sigma[0.0] = {
            "baseline_f1": ref_base,
            "drops": ref_drops,
            "rank": ref_rank,
            "spearman_vs_ref": 1.0,
        }
        for s in sigma_list[1:]:
            drops, base = _part_ranking_under_noise(
                model, loader, dev, noise_sigma=s, seed=seed + f_id + int(s * 1e5)
            )
            rank = _rank_order(drops)
            rho = _spearman(ref_rank, rank)
            per_sigma[s] = {
                "baseline_f1": base,
                "drops": drops,
                "rank": rank,
                "spearman_vs_ref": rho,
            }
            click.echo(f"  σ={s:.3f} baseline_f1={base:.4f} rank={rank} ρ={rho:+.3f}")

        payload = {
            "experiment": experiment,
            "fold": f_id,
            "parts_order": PARTS_ORDERED,
            "sigmas": sigma_list,
            "per_sigma": per_sigma,
        }
        out_json = out_root / f"{experiment}_fold{f_id:02d}_stability.json"
        out_json.write_text(json.dumps(payload, indent=2))

        for s, d in per_sigma.items():
            all_records.append({
                "experiment": experiment,
                "fold": f_id,
                "sigma": s,
                "baseline_f1": d["baseline_f1"],
                "spearman_vs_ref": d["spearman_vs_ref"],
                **{f"drop_{p}": d["drops"][p] for p in PARTS_ORDERED},
            })

    df = pd.DataFrame(all_records)
    csv_path = out_root / f"{experiment}_stability.csv"
    df.to_csv(csv_path, index=False)
    click.echo(f"\nSaved: {csv_path}  ({len(df)} rows)")

    # Simple summary: mean ρ per sigma across folds
    if len(df) > 0:
        agg = df.groupby("sigma")["spearman_vs_ref"].agg(["mean", "std"]).round(4)
        click.echo("\nSpearman rank correlation vs σ=0 (mean ± std across folds):")
        click.echo(str(agg))


if __name__ == "__main__":
    main()
