""" body-part masking faithfulness evaluation.

See: (exp058)
Related:
- tools/infer_test_ensemble.py (checkpoint loading pattern)
- diema/models/skeleton_graph.py::DIEMA_BODY_PARTS (6-group partition)
- diema/training/callbacks.py::OOFPredictionCallback (baseline val logits)

For a trained experiment, this tool:

1. Rebuilds the model from ``config.yaml`` and loads the best.ckpt.
2. Runs val-fold inference under each of 7 inputs: the unmasked baseline and
   6 body-part zero-mask variants (``torso``/``head``/``r_arm``/``l_arm``/
   ``r_leg``/``l_leg``). Masking means setting ``x[:, :, :, node] = 0`` for
   every joint index in ``DIEMA_BODY_PARTS[part]``.
3. Records per-part macro-F1 and per-class F1 drops vs the baseline.
4. Computes three masking orderings for ERASER-style faithfulness AUCs:
   - ``important_first``: largest-F1-drop part first
   - ``reverse``:          smallest-F1-drop part first
   - ``random``:           null baseline (seeded)
   For each ordering, cumulatively mask top-k (k=0..6) parts and record F1.
5. Writes a CSV (``part_importance_<exp>_fold<N>.csv``) + heatmap PNG
   (emotion × part F1 drop) to the artifacts dir.

Usage::

    python tools/explain_part_masking.py \\
        --experiment exp034_regionaware_convtr_a00 \\
        --fold 0 \\
        --device cuda

    # Aggregate all folds, average the drops:
    python tools/explain_part_masking.py \\
        --experiment exp034_regionaware_convtr_a00 \\
        --fold all
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
from diema.data.splits import generate_lpo_splits  # noqa: E402

from tools.infer_test_ensemble import _cfg_to_builder_ns, _load_checkpoint_into_model  # noqa: E402


PARTS_ORDERED = list(DIEMA_BODY_PARTS.keys())  # stable order
NUM_CLASS = 12


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    from sklearn.metrics import f1_score
    return np.asarray(
        f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASS)), zero_division=0)
    )


@torch.no_grad()
def _forward_with_mask(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    mask_nodes: tuple[int, ...] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run validation inference with an optional joint-index zero mask.

    Returns ``(logits, labels)`` numpy arrays.
    """
    model = model.to(device).eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for batch in loader:
        x, y, _fn = batch
        # ``x.to(device)`` is a no-op when the batch is already on ``device``
        # (e.g. CPU-only loaders), returning the same underlying storage.
        # Mutating it in-place would corrupt the DataLoader's persistent
        # buffers and poison subsequent forward passes on the same val loop.
        # ``.clone()`` guarantees we never write into the loader's memory.
        x = x.to(device).clone()
        if mask_nodes is not None:
            for n in mask_nodes:
                x[:, :, :, n] = 0.0
        out = model(x)
        logits = out["logits"] if isinstance(out, dict) else out
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.numpy())
    if not all_logits:
        # Empty val (e.g. --country JP|TW on a fold with no matching performer).
        # Returning typed-but-empty arrays lets the caller treat this as a
        # no-op fold without special-casing len() == 0 at every call site.
        return (
            np.empty((0, NUM_CLASS), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.concatenate(all_logits, 0), np.concatenate(all_labels, 0)


def _build_val_loader(cfg_dict: dict, fold: int, num_folds: int,
                      batch_size: int = 128,
                      country: str | None = None,
                      intensity: str | None = None) -> torch.utils.data.DataLoader:
    """Construct a deterministic val-fold DataLoader from the saved config.

    Args:
        country: stratification filter. When set to "JP" or
            ``"TW"``, only val samples whose filename starts with that prefix
            are kept. ``None`` keeps the full val fold.
        intensity: stratification filter. When set to "L", "M",
            or ``"H"``, only val samples whose DIEM-A filename intensity suffix
            matches are kept (``..._{scenario}_{intensity}`` convention).
    """
    from diema.data.dataset import MotionDataset
    from torch.utils.data import DataLoader

    env = EnvConfig()
    data_path = cfg_dict.get("data_path", "")
    if data_path:
        path = Path(data_path)
        if not path.is_absolute():
            path = env.project_root / path
    else:
        path = env.processed_dir / "motion_train_quat.npz"
    if not path.exists():
        raise FileNotFoundError(f"train NPZ not found: {path}")

    preprocessed = pybvh_ml.load_preprocessed(str(path))
    filenames = preprocessed["filenames"]
    splits = generate_lpo_splits(filenames, num_folds=num_folds)
    split_dict = splits[fold]
    val_indices = [idx for _, idx in split_dict["val"]]

    if country is not None:
        if country not in ("JP", "TW"):
            raise ValueError(f"country must be 'JP' or 'TW', got {country!r}")
        prefix = f"{country}_"
        val_indices = [i for i in val_indices if Path(filenames[i]).stem.startswith(prefix)]

    if intensity is not None:
        if intensity not in ("L", "M", "H"):
            raise ValueError(f"intensity must be 'L', 'M', or 'H', got {intensity!r}")
        # DIEM-A filename = {country}_{id}_{emotion}_{scenario}_{intensity}[_augNN]
        # Match the intensity field as the LAST "_"-delimited segment of the stem
        # (ignoring any _augNN suffix). We parse via parse_filename when available
        # and fall back to a tolerant stem-split so tests can run on synthetic names.
        from diema.data.parser import parse_filename
        kept: list[int] = []
        for i in val_indices:
            stem = Path(filenames[i]).stem
            try:
                info = parse_filename(stem)
                if info.intensity == intensity:
                    kept.append(i)
            except Exception:
                # Fallback: last underscore-delimited chunk (strip _augNN if present)
                parts = stem.split("_")
                if parts and parts[-1].startswith("aug"):
                    parts = parts[:-1]
                if parts and parts[-1] == intensity:
                    kept.append(i)
        val_indices = kept

    ds = MotionDataset(
        data_path=str(path),
        indices=val_indices,
        clip_length=cfg_dict.get("clip_length", 64),
        target_repr=cfg_dict.get("target_repr", "6d"),
        is_test=True,
        augmentation_pipeline=None,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=False)


def _faithfulness_curve(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    order: list[str],
    labels_baseline: np.ndarray,
) -> list[float]:
    """Cumulatively mask parts in ``order``, return F1 at k=0..len(order)."""
    curve: list[float] = []
    pred0 = None
    for k in range(len(order) + 1):
        if k == 0:
            mask_nodes = None
        else:
            nodes: list[int] = []
            for p in order[:k]:
                nodes.extend(DIEMA_BODY_PARTS[p])
            mask_nodes = tuple(sorted(set(nodes)))
        logits, labels = _forward_with_mask(model, loader, device, mask_nodes)
        if pred0 is None:
            # Baseline: remember label ordering for later sanity
            assert np.array_equal(labels, labels_baseline), (
                "val label ordering changed between baseline and masked runs"
            )
            pred0 = logits.argmax(1)
        f1 = _macro_f1(labels, logits.argmax(1))
        curve.append(f1)
    return curve


def _auc_drop(curve: list[float]) -> float:
    """Trapezoidal area between the baseline (curve[0]) and the rest.

    Normalised by the number of masking steps so it is a scalar in [0, 1]
    (approximately) — larger = faster F1 drop = more faithful attribution
    (for important_first) or less faithful (for reverse/random).
    """
    base = curve[0]
    drops = [base - v for v in curve[1:]]
    if not drops:
        return 0.0
    # Simple mean of cumulative drops ≈ area under the drop curve.
    return float(np.mean(drops))


@click.command()
@click.option("--experiment", required=True, help="Experiment folder name under output/artifacts/")
@click.option("--fold", default="0", help="Fold number or 'all' to aggregate across folds")
@click.option("--num-folds", default=10, type=int)
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]))
@click.option("--batch-size", default=128, type=int)
@click.option("--country", default=None, type=click.Choice([None, "JP", "TW"]),
              help=" stratify val by country (prefix match on filename)")
@click.option("--intensity", default=None, type=click.Choice([None, "L", "M", "H"]),
              help=" stratify val by DIEM-A intensity tag (filename suffix)")
@click.option("--output-dir", default=None,
              help="Where to save CSV + figure. Defaults to docs/analysis/part_masking/.")
def main(experiment: str, fold: str, num_folds: int, device: str,
         batch_size: int, country: str | None, intensity: str | None,
         output_dir: str | None) -> None:
    env = EnvConfig()
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    click.echo(f"Device: {dev}")

    exp_dir = env.artifacts_dir / experiment
    if fold == "all":
        fold_ids = [int(p.name.split("_")[1]) for p in sorted(exp_dir.glob("fold_[0-9][0-9]"))]
    else:
        fold_ids = [int(fold)]

    out_root = Path(output_dir) if output_dir else (env.project_root / "docs/analysis/part_masking")
    out_root.mkdir(parents=True, exist_ok=True)

    all_fold_records: list[dict] = []
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
                                   batch_size=batch_size, country=country,
                                   intensity=intensity)

        # 1. Baseline val-fold logits.
        base_logits, labels = _forward_with_mask(model, loader, dev, None)
        if labels.size == 0:
            suffix = f" (country={country})" if country else ""
            click.echo(
                f"[skip] fold_{f_id:02d}{suffix}: 0 val samples after filtering",
                err=True,
            )
            continue
        base_pred = base_logits.argmax(1)
        base_f1 = _macro_f1(labels, base_pred)
        base_per_class = _per_class_f1(labels, base_pred)
        click.echo(f"\nfold_{f_id:02d} baseline Macro-F1: {base_f1:.4f}")

        # 2. Per-part zero mask → F1 drop.
        part_f1: dict[str, float] = {}
        part_per_class: dict[str, np.ndarray] = {}
        for p in PARTS_ORDERED:
            logits_p, _ = _forward_with_mask(
                model, loader, dev, tuple(DIEMA_BODY_PARTS[p]),
            )
            pred_p = logits_p.argmax(1)
            part_f1[p] = _macro_f1(labels, pred_p)
            part_per_class[p] = _per_class_f1(labels, pred_p)
            click.echo(f"  mask {p:>6}: Macro-F1 {part_f1[p]:.4f} "
                       f"(drop {base_f1 - part_f1[p]:+.4f})")

        # 3. Faithfulness curves (3 orderings).
        drops_by_part = {p: base_f1 - part_f1[p] for p in PARTS_ORDERED}
        order_important = sorted(PARTS_ORDERED, key=lambda p: -drops_by_part[p])
        order_reverse = list(reversed(order_important))
        rng = np.random.default_rng(f_id)  # deterministic per fold
        order_random = list(PARTS_ORDERED)
        rng.shuffle(order_random)

        curves = {
            "important_first": _faithfulness_curve(model, loader, dev,
                                                    order_important, labels),
            "reverse": _faithfulness_curve(model, loader, dev, order_reverse, labels),
            "random": _faithfulness_curve(model, loader, dev, order_random, labels),
        }
        aucs = {name: _auc_drop(c) for name, c in curves.items()}
        click.echo(
            f"  faithfulness AUC — important_first: {aucs['important_first']:.4f}, "
            f"reverse: {aucs['reverse']:.4f}, random: {aucs['random']:.4f}"
        )

        # 4. Record.
        for p in PARTS_ORDERED:
            row = {
                "experiment": experiment,
                "fold": f_id,
                "country": country or "ALL",
                "intensity": intensity or "ALL",
                "part": p,
                "baseline_f1": base_f1,
                "masked_f1": part_f1[p],
                "f1_drop": base_f1 - part_f1[p],
            }
            for cls_idx in range(NUM_CLASS):
                row[f"f1_drop_{IDX_TO_EMOTION[cls_idx]}"] = float(
                    base_per_class[cls_idx] - part_per_class[p][cls_idx]
                )
            all_fold_records.append(row)

        # Write per-fold JSON for faithfulness curves.
        curve_payload = {
            "experiment": experiment,
            "fold": f_id,
            "country": country,
            "baseline_f1": base_f1,
            "order_important": order_important,
            "order_reverse": order_reverse,
            "order_random": order_random,
            "curves": curves,
            "aucs": aucs,
        }
        stratum_parts: list[str] = []
        if country:
            stratum_parts.append(country)
        if intensity:
            stratum_parts.append(f"int{intensity}")
        country_suffix = ("_" + "_".join(stratum_parts)) if stratum_parts else ""
        out_curve = out_root / f"{experiment}_fold{f_id:02d}{country_suffix}_faithfulness.json"
        out_curve.write_text(json.dumps(curve_payload, indent=2))
        click.echo(f"  saved faithfulness JSON: {out_curve}")

    # 5. Aggregate CSV.
    df = pd.DataFrame(all_fold_records)
    stratum_parts: list[str] = []
    if country:
        stratum_parts.append(country)
    if intensity:
        stratum_parts.append(f"int{intensity}")
    agg_suffix = ("_" + "_".join(stratum_parts)) if stratum_parts else ""
    csv_path = out_root / f"{experiment}_part_importance{agg_suffix}.csv"
    df.to_csv(csv_path, index=False)
    click.echo(f"\nSaved aggregated CSV: {csv_path}  ({len(df)} rows)")

    # 6. Optional heatmap figure (avg across folds if fold=='all').
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        agg = df.groupby("part").mean(numeric_only=True)
        drop_cols = [c for c in agg.columns if c.startswith("f1_drop_")]
        heatmap = agg[drop_cols].reindex(PARTS_ORDERED)
        heatmap.columns = [c.replace("f1_drop_", "") for c in heatmap.columns]
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(heatmap.values, aspect="auto", cmap="RdYlBu_r")
        ax.set_xticks(range(len(heatmap.columns)))
        ax.set_xticklabels(heatmap.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(PARTS_ORDERED)))
        ax.set_yticklabels(PARTS_ORDERED)
        stratum_label = ""
        if stratum_parts:
            stratum_label = f" ({', '.join(stratum_parts)})"
        ax.set_title(f"F1 drop when masking each body-part — {experiment}{stratum_label}")
        fig.colorbar(im, ax=ax, label="F1 drop (baseline - masked)")
        fig.tight_layout()
        fig_path = out_root / f"{experiment}_part_heatmap{agg_suffix}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        click.echo(f"Saved heatmap: {fig_path}")
    except Exception as exc:
        click.echo(f"[warn] heatmap skipped: {exc}", err=True)


if __name__ == "__main__":
    main()
