""" ensemble part-masking faithfulness for the submitted ensemble.

Related:
- tools/explain_part_masking.py
    Single-model baseline that this generalises. Reuses ``_build_val_loader``,
    ``_forward_with_mask`` semantics, ``_auc_drop``.
- experiments/exp090_ensemble_patterns/build_logitmean_submission.py
    Canonical logit-mean fusion that the submitted ensemble uses.
- tools/infer_test_ensemble.py
    Source of ``_cfg_to_builder_ns`` / ``_load_checkpoint_into_model``.
- diema/models/skeleton_graph.py::DIEMA_BODY_PARTS
    Joint-index partition for 6 body parts.

What this does
==============

The submitted 11-way logit-mean ensemble is reviewed for *faithfulness*:
when we zero out the input joints of a body part, does the ensemble's output
change in a way consistent with our explainability claim?

For each LPO fold and each body part (``torso, head, r_arm, l_arm, r_leg,
l_leg``), this tool:

1.  Loads val samples for the fold (same LPO split as training).
2.  Runs all PROD ensemble members on the *unmasked* input → records each
    member's logits and the logit-mean ensemble baseline Macro-F1.
3.  Runs all members on the *single-part masked* input (one part at a time)
    → records each per-part ensemble F1 and drop = baseline - masked.
4.  Computes ERASER-style faithfulness AUCs under three cumulative
    orderings — ``important_first`` (largest drops first), ``reverse``,
    and ``random`` (seeded null). For each k=0..6 we cumulatively mask
    the top-k parts in the order, run *all members again*, fuse, and
    record ensemble F1.
5.  Writes per-fold JSON + aggregated CSV + emotion×part heatmap.

Logit-mean fusion (matches the submission script)::

    P = softmax(logits)        # (M, N, C) per-member probabilities
    fused_logprob = log(P).mean(0)
    final_pred = fused_logprob.argmax(1)   # softmax is monotone, omit

 (this MVP) covers the 7 PROD members:
``exp002_a04_smooth, exp003_ctrgcn_a00, exp004_skateformer_a01,
exp006_protogcn_a00, exp020_conv1d_transformer_a01,
exp023_keypoint_pool_mlp_a00, exp034_regionaware_convtr_a00``.

 extends to skeleton-based externals (MotionBERT, MAMP-60xsub,
MAMP-120xset) by re-extracting their frozen features under the same
masked input. adds C3D via marker-level masking.

Usage::

    python tools/explain_ensemble_part_masking.py --fold 0      # smoke
    python tools/explain_ensemble_part_masking.py --fold all    # full

Outputs (``docs/analysis/ensemble_part_masking/``):

- ``ensemble_7way_fold{NN}_faithfulness.json``
- ``ensemble_7way_part_importance.csv``
- ``ensemble_7way_part_heatmap.png``
- ``ensemble_7way_member_baseline_logits/fold_{NN}/{exp}.npy``  (cache)
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

import pybvh_ml  # noqa: E402,F401
from utils.env import EnvConfig  # noqa: E402
from diema.models import build_model  # noqa: E402
import diema.models  # noqa: F401,E402  (register models)
from diema.models.skeleton_graph import DIEMA_BODY_PARTS  # noqa: E402
from diema.data.parser import IDX_TO_EMOTION  # noqa: E402

from tools.infer_test_ensemble import (  # noqa: E402
    _cfg_to_builder_ns,
    _load_checkpoint_into_model,
)
from tools.explain_part_masking import (  # noqa: E402
    _auc_drop,
    _build_val_loader,
    _macro_f1,
    _per_class_f1,
)


PARTS_ORDERED: list[str] = list(DIEMA_BODY_PARTS.keys())  # stable order
NUM_CLASS = 12

# Canonical 7 PROD members (must mirror
# experiments/exp090_ensemble_patterns/build_logitmean_submission.py::MEMBERS).
PROD_MEMBERS: tuple[str, ...] = (
    "exp002_a04_smooth",
    "exp003_ctrgcn_a00",
    "exp004_skateformer_a01",
    "exp006_protogcn_a00",
    "exp020_conv1d_transformer_a01",
    "exp023_keypoint_pool_mlp_a00",
    "exp034_regionaware_convtr_a00",
)


# --------------------------------------------------------------------------- #
# Fusion math
# --------------------------------------------------------------------------- #


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _logit_mean_fuse(member_logits: list[np.ndarray]) -> np.ndarray:
    """Canonical logit-mean fusion = mean of per-member log-softmax.

    Mirrors the submission script's
    ``softmax(np.log(np.clip(P, 1e-12, 1.0)).mean(axis=0))`` minus the
    outer softmax (which is monotone and not needed for argmax).
    """
    if len(member_logits) == 0:
        raise ValueError("member_logits must not be empty")
    # Validate consistent shape; numpy's stack raises ValueError, but be explicit.
    shape0 = member_logits[0].shape
    for i, lg in enumerate(member_logits):
        if lg.shape != shape0:
            raise ValueError(
                f"member {i} shape {lg.shape} != member 0 shape {shape0}"
            )
    L = np.stack(member_logits, axis=0).astype(np.float64)  # (M, N, C)
    P = _softmax(L, axis=-1)
    return np.log(np.clip(P, 1e-12, 1.0)).mean(axis=0)


def _logit_mean_fuse_argmax(member_logits: list[np.ndarray]) -> np.ndarray:
    """Ensemble argmax — same as ``_logit_mean_fuse(...).argmax(1)``."""
    return _logit_mean_fuse(member_logits).argmax(axis=1)


def _ensemble_macro_f1(member_logits: list[np.ndarray], labels: np.ndarray) -> float:
    pred = _logit_mean_fuse_argmax(member_logits)
    return _macro_f1(labels, pred)


# --------------------------------------------------------------------------- #
# Mask key plumbing
# --------------------------------------------------------------------------- #


def _mask_key(parts: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """Canonicalise a mask spec to a sorted, deduped tuple of part names."""
    if parts is None or len(parts) == 0:
        return ()
    return tuple(sorted(set(parts)))


def _mask_nodes_for_key(key: tuple[str, ...]) -> tuple[int, ...]:
    """Joint indices for a mask key (sorted unique union of part members)."""
    nodes: list[int] = []
    for p in key:
        nodes.extend(DIEMA_BODY_PARTS[p])
    return tuple(sorted(set(nodes)))


def _apply_mask_inplace(
    x: torch.Tensor,
    mask_nodes: tuple[int, ...],
    mode: str,
    perm: torch.Tensor | None = None,
) -> None:
    """Apply the chosen perturbation to ``x[:, :, :, mask_nodes]`` in place.

    Modes
    -----
    ``zero``:
        Replace the masked joints' values with ``0``. The original probe
        from the single-model tool. Aggressive: empirically collapses 6/7
        ensemble members to a single-class majority prediction (because
        non-Region-Aware models are not trained to tolerate zero joints).
        Useful as a destructive upper bound but a poor probe for the
        ensemble's true reliance on a body part.
    ``shuffle``:
        Permutation-importance style. Replace ``x[b, :, :, n]`` with
        ``x[perm[b], :, :, n]``, where ``perm`` is a batch-wise random
        permutation (seeded per fold for determinism). Preserves the
        marginal distribution of each joint (no out-of-distribution
        artifacts) while destroying the *individual* information content
        of that joint for that sample. Standard probe for permutation
        feature importance (Breiman 2001; Fisher et al. 2019). Mask is
        graceful for all members, giving a meaningful ensemble gap.

    Note
    ----
    ``perm`` is required iff ``mode == 'shuffle'``. It must be a
    1-D LongTensor of length ``x.shape[0]`` on the same device.
    """
    if mode == "zero":
        for n in mask_nodes:
            x[:, :, :, n] = 0.0
    elif mode == "shuffle":
        if perm is None:
            raise ValueError("shuffle mask requires a permutation tensor")
        if perm.shape[0] != x.shape[0]:
            raise ValueError(
                f"perm has length {perm.shape[0]} but batch is {x.shape[0]}"
            )
        # In-place copy of the permuted joint slice.
        for n in mask_nodes:
            x[:, :, :, n] = x.index_select(0, perm)[:, :, :, n]
    else:
        raise ValueError(f"unknown mask mode: {mode!r} (use 'zero' or 'shuffle')")


def _orderings_for_drops(
    drops: dict[str, float], seed: int
) -> tuple[list[str], list[str], list[str]]:
    """Build the 3 orderings used for the faithfulness curve.

    ``important_first`` sorts by drop magnitude (largest first; ties broken
    by alphabetical part name for determinism). ``reverse`` reverses it.
    ``random`` is a seeded shuffle of ``PARTS_ORDERED``.
    """
    important = sorted(PARTS_ORDERED, key=lambda p: (-drops[p], p))
    reverse = list(reversed(important))
    rng = np.random.default_rng(seed)
    random_order = list(PARTS_ORDERED)
    rng.shuffle(random_order)
    return important, reverse, random_order


# --------------------------------------------------------------------------- #
# Per-member inference with cached masks
# --------------------------------------------------------------------------- #


@torch.no_grad()
def _run_member_for_mask_set(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    mask_keys: list[tuple[str, ...]],
    mask_mode: str = "zero",
    perm_seed: int = 0,
    batch_size: int = 128,
) -> tuple[dict[tuple[str, ...], np.ndarray], np.ndarray]:
    """Run one model forward for every mask in ``mask_keys``.

    Returns ``(logits_per_key, labels)``. ``logits_per_key[key]`` has shape
    ``(N_val, NUM_CLASS)``; ``labels`` has shape ``(N_val,)``.

    Protocol (post-C1 fix, 2026-06-12):
    1. Accumulate the entire val fold into a single CPU tensor ``X``
       (so a single global permutation can be applied at val-set scope,
       matching Breiman 2001 / Fisher et al. 2019 permutation-importance
       semantics and matching the external members' protocol in
       ``explain_external_part_masking.py``).
    2. Build one global ``perm`` of length ``N_val`` (seeded by
       ``perm_seed`` for reproducibility). The same ``perm`` is reused
       across all mask keys for fairness — so the per-mask Macro-F1 drops
       are directly comparable.
    3. For each mask key, apply the mask to a copy of ``X`` and run the
       model forward in mini-batches.

    Memory cost is bounded by ``X.shape == (N_val, C, T, V)`` which for
    the largest DIEM-A fold (n=864, C=6, T=64, V=25) is ~33 MB per
    tensor — trivially fits in GPU memory even alongside the model.
    """
    model = model.to(device).eval()

    mask_nodes: dict[tuple[str, ...], tuple[int, ...]] = {
        key: _mask_nodes_for_key(key) for key in mask_keys
    }
    buffers: dict[tuple[str, ...], list[np.ndarray]] = {key: [] for key in mask_keys}

    # --- 1. Accumulate the whole val fold so the permutation is val-set wide ---
    x_chunks: list[torch.Tensor] = []
    labels_chunks: list[np.ndarray] = []
    for batch in loader:
        x, y, _fn = batch
        x_chunks.append(x)
        labels_chunks.append(y.numpy())

    if not labels_chunks:
        empty = {key: np.empty((0, NUM_CLASS), dtype=np.float32) for key in mask_keys}
        return empty, np.empty((0,), dtype=np.int64)

    X_cpu = torch.cat(x_chunks, dim=0)
    labels = np.concatenate(labels_chunks, axis=0)
    N_val = X_cpu.shape[0]

    # --- 2. One global permutation, shared across mask keys for fairness ---
    perm_dev: torch.Tensor | None = None
    if mask_mode == "shuffle":
        rng = np.random.default_rng(perm_seed)
        perm_dev = torch.from_numpy(rng.permutation(N_val)).to(device)

    # --- 3. For each mask key, mask the whole val tensor and forward in batches ---
    for key in mask_keys:
        # Move the un-masked val set to device for this key; clone for masking.
        X_dev = X_cpu.to(device).clone()
        if mask_nodes[key]:
            _apply_mask_inplace(X_dev, mask_nodes[key], mask_mode, perm=perm_dev)
        # Mini-batch the (already-masked) tensor through the model.
        for s in range(0, N_val, batch_size):
            e = min(s + batch_size, N_val)
            out = model(X_dev[s:e])
            logits = out["logits"] if isinstance(out, dict) else out
            buffers[key].append(logits.detach().cpu().numpy())
        # Free GPU memory before the next key.
        del X_dev
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_logits = {key: np.concatenate(buffers[key], axis=0) for key in mask_keys}
    return out_logits, labels


# --------------------------------------------------------------------------- #
# Fold driver
# --------------------------------------------------------------------------- #


def _resolve_mask_keys_for_curves(part_drops: dict[str, float], seed: int,
                                  ) -> tuple[list[tuple[str, ...]],
                                             list[str], list[str], list[str]]:
    """All unique mask-key sets we need to evaluate for the 3 curves.

    Includes baseline ``()`` and all cumulative top-k unions for each
    ordering. Returns ``(keys, important, reverse, random)`` where the
    orderings are the part-name lists.
    """
    important, reverse, random_order = _orderings_for_drops(part_drops, seed)
    keys: set[tuple[str, ...]] = {()}  # baseline
    for order in (important, reverse, random_order):
        for k in range(1, len(order) + 1):
            keys.add(_mask_key(tuple(order[:k])))
    return sorted(keys, key=lambda t: (len(t), t)), important, reverse, random_order


def _faithfulness_curve_from_cache(
    fused_f1_by_key: dict[tuple[str, ...], float],
    order: list[str],
) -> list[float]:
    """Cumulative F1 at k=0..6 by reading the prefilled cache."""
    curve: list[float] = []
    for k in range(len(order) + 1):
        if k == 0:
            key = ()
        else:
            key = _mask_key(tuple(order[:k]))
        curve.append(fused_f1_by_key[key])
    return curve


def _run_fold(
    fold_id: int,
    num_folds: int,
    members: tuple[str, ...],
    env: EnvConfig,
    device: torch.device,
    batch_size: int,
    country: str | None,
    intensity: str | None,
    out_root: Path,
    mask_mode: str = "zero",
) -> dict | None:
    """Run all ensemble masking measurements for one fold."""
    # ---- 1. Per-member: load model + run baseline + single-part masks ----
    val_loader = None
    cfg_seed: dict | None = None

    # Single-part mask keys are computed up-front; we extend with cumulative
    # keys after we know the drop ordering.
    single_keys: list[tuple[str, ...]] = [()] + [(p,) for p in PARTS_ORDERED]

    per_member_baseline_logits: dict[str, np.ndarray] = {}
    per_member_single_logits: dict[str, dict[tuple[str, ...], np.ndarray]] = {}
    labels_baseline: np.ndarray | None = None

    for exp_name in members:
        fold_dir = env.artifacts_dir / exp_name / f"fold_{fold_id:02d}"
        ckpt = fold_dir / "best.ckpt"
        cfg_path = fold_dir / "config.yaml"
        if not (ckpt.exists() and cfg_path.exists()):
            click.echo(
                f"[skip-fold] fold_{fold_id:02d}: missing ckpt/config for {exp_name}",
                err=True,
            )
            return None

        cfg_dict = OmegaConf.to_container(OmegaConf.load(str(cfg_path)), resolve=True)
        if cfg_seed is None:
            cfg_seed = cfg_dict
            # Build the val loader from the FIRST member's config; all PROD
            # members share clip_length / target_repr / 25-node CTV input,
            # so the same DataLoader feeds all of them. We assert this on
            # the spot for any drift.
            val_loader = _build_val_loader(
                cfg_dict, fold=fold_id, num_folds=num_folds,
                batch_size=batch_size, country=country, intensity=intensity,
            )
        else:
            for k in ("clip_length", "target_repr"):
                assert cfg_dict.get(k) == cfg_seed.get(k), (
                    f"{exp_name} fold_{fold_id} disagrees with member 0 on {k}: "
                    f"{cfg_dict.get(k)} vs {cfg_seed.get(k)}"
                )

        builder_ns = _cfg_to_builder_ns(cfg_dict)
        model = build_model(builder_ns)
        _load_checkpoint_into_model(ckpt, model)

        click.echo(f"  [member] fold_{fold_id:02d} {exp_name} ({mask_mode}) …")
        # Per-fold permutation seed so shuffle is deterministic across runs
        # but independent across folds. Members within a fold share the
        # same global permutation (one per val set, post-C1 fix).
        per_mask, labels = _run_member_for_mask_set(
            model, val_loader, device, single_keys,
            mask_mode=mask_mode, perm_seed=fold_id * 1000,
            batch_size=batch_size,
        )
        if labels.size == 0:
            suffix = f" (country={country}, intensity={intensity})" \
                if (country or intensity) else ""
            click.echo(
                f"[skip-fold] fold_{fold_id:02d}{suffix}: 0 val samples after filtering",
                err=True,
            )
            del model
            return None

        if labels_baseline is None:
            labels_baseline = labels
        else:
            assert np.array_equal(labels, labels_baseline), (
                f"label ordering drift between {exp_name} and the first member; "
                "OOF filenames should be identical across PROD members"
            )

        per_member_baseline_logits[exp_name] = per_mask[()]
        per_member_single_logits[exp_name] = {
            k: per_mask[k] for k in single_keys if k != ()
        }

        # Free GPU memory before the next member.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert labels_baseline is not None

    # ---- 2. Ensemble baseline + single-part Macro-F1 ----
    ensemble_baseline_logits = [per_member_baseline_logits[m] for m in members]
    base_f1 = _ensemble_macro_f1(ensemble_baseline_logits, labels_baseline)
    base_per_class = _per_class_f1(
        labels_baseline, _logit_mean_fuse_argmax(ensemble_baseline_logits)
    )
    click.echo(
        f"\nfold_{fold_id:02d} ensemble (7-way PROD) baseline Macro-F1: {base_f1:.4f}"
    )

    part_f1: dict[str, float] = {}
    part_per_class: dict[str, np.ndarray] = {}
    fused_f1_by_key: dict[tuple[str, ...], float] = {(): base_f1}
    for p in PARTS_ORDERED:
        key = (p,)
        member_logits = [per_member_single_logits[m][key] for m in members]
        pred = _logit_mean_fuse_argmax(member_logits)
        part_f1[p] = _macro_f1(labels_baseline, pred)
        part_per_class[p] = _per_class_f1(labels_baseline, pred)
        fused_f1_by_key[key] = part_f1[p]
        click.echo(
            f"  mask {p:>6}: ensemble Macro-F1 {part_f1[p]:.4f} "
            f"(drop {base_f1 - part_f1[p]:+.4f})"
        )

    # ---- 3. Cumulative masks for faithfulness curves ----
    drops_by_part = {p: base_f1 - part_f1[p] for p in PARTS_ORDERED}
    cumulative_keys, important, reverse, random_order = (
        _resolve_mask_keys_for_curves(drops_by_part, seed=fold_id)
    )
    needed = [k for k in cumulative_keys if k not in fused_f1_by_key]

    if needed:
        click.echo(
            f"  cumulative masks: {len(needed)} new sets (out of "
            f"{len(cumulative_keys)} total) — running each member again"
        )
        per_member_cum_logits: dict[str, dict[tuple[str, ...], np.ndarray]] = {}
        for exp_name in members:
            fold_dir = env.artifacts_dir / exp_name / f"fold_{fold_id:02d}"
            cfg_dict = OmegaConf.to_container(
                OmegaConf.load(str(fold_dir / "config.yaml")), resolve=True
            )
            builder_ns = _cfg_to_builder_ns(cfg_dict)
            model = build_model(builder_ns)
            _load_checkpoint_into_model(fold_dir / "best.ckpt", model)
            click.echo(f"  [cum] fold_{fold_id:02d} {exp_name} ({mask_mode}) …")
            per_mask, _ = _run_member_for_mask_set(
                model, val_loader, device, needed,
                mask_mode=mask_mode, perm_seed=fold_id * 1000,
                batch_size=batch_size,
            )
            per_member_cum_logits[exp_name] = per_mask
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for key in needed:
            member_logits = [per_member_cum_logits[m][key] for m in members]
            fused_f1_by_key[key] = _ensemble_macro_f1(member_logits, labels_baseline)

    curves = {
        "important_first": _faithfulness_curve_from_cache(fused_f1_by_key, important),
        "reverse": _faithfulness_curve_from_cache(fused_f1_by_key, reverse),
        "random": _faithfulness_curve_from_cache(fused_f1_by_key, random_order),
    }
    aucs = {name: _auc_drop(c) for name, c in curves.items()}
    click.echo(
        f"  faithfulness AUC — important_first: {aucs['important_first']:.4f}, "
        f"reverse: {aucs['reverse']:.4f}, random: {aucs['random']:.4f}, "
        f"gap (important-reverse): {aucs['important_first']-aucs['reverse']:+.4f}"
    )

    # ---- 4. Per-fold record + JSON ----
    fold_record = {
        "ensemble": "7way_prod_logitmean",
        "mask_mode": mask_mode,
        "members": list(members),
        "fold": fold_id,
        "country": country,
        "intensity": intensity,
        "baseline_f1": base_f1,
        "part_f1": part_f1,
        "drops_by_part": drops_by_part,
        "order_important": important,
        "order_reverse": reverse,
        "order_random": random_order,
        "curves": curves,
        "aucs": aucs,
        "n_val": int(labels_baseline.shape[0]),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    stratum_parts: list[str] = [mask_mode]
    if country:
        stratum_parts.append(country)
    if intensity:
        stratum_parts.append(f"int{intensity}")
    suffix = "_" + "_".join(stratum_parts)
    out_curve = out_root / f"ensemble_7way_fold{fold_id:02d}{suffix}_faithfulness.json"
    out_curve.write_text(json.dumps(fold_record, indent=2))
    click.echo(f"  saved per-fold JSON: {out_curve}")

    # Also save per-member baseline logits so downstream (LMA alignment,
    # bootstrap CI) can re-fuse without re-running models.
    cache_dir = out_root / "member_baseline_logits" / f"fold_{fold_id:02d}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for exp_name, lg in per_member_baseline_logits.items():
        np.save(cache_dir / f"{exp_name}.npy", lg.astype(np.float32))
    np.save(cache_dir / "_labels.npy", labels_baseline.astype(np.int64))

    # And save per-member per-mask logits (for downstream 10-way/11-way
    # ensemble fusion combining with external members' masked logits).
    masked_cache_dir = (
        out_root / "member_masked_logits" / "prod"
        / f"fold_{fold_id:02d}_{mask_mode}"
    )
    masked_cache_dir.mkdir(parents=True, exist_ok=True)
    # Build a unified {mask_key: {member: logits}} from all sources.
    all_masked_logits: dict[tuple[str, ...], dict[str, np.ndarray]] = {}
    for member in members:
        all_masked_logits.setdefault((), {})[member] = per_member_baseline_logits[member]
        for key, lg in per_member_single_logits[member].items():
            all_masked_logits.setdefault(key, {})[member] = lg
        if needed:
            for key, lg in per_member_cum_logits.get(member, {}).items():
                all_masked_logits.setdefault(key, {})[member] = lg
    # Persist per (mask_key, member) tensor.
    for key, by_member in all_masked_logits.items():
        key_name = "baseline" if len(key) == 0 else "_".join(sorted(key))
        for member, lg in by_member.items():
            np.save(masked_cache_dir / f"{key_name}__{member}.npy", lg.astype(np.float32))
    np.save(masked_cache_dir / "_labels.npy", labels_baseline.astype(np.int64))

    # Per-emotion drops table rows for the aggregated CSV.
    rows: list[dict] = []
    for p in PARTS_ORDERED:
        row: dict = {
            "ensemble": "7way_prod_logitmean",
            "mask_mode": mask_mode,
            "fold": fold_id,
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
        rows.append(row)
    fold_record["_csv_rows"] = rows
    return fold_record


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@click.command()
@click.option("--fold", default="0", help="Fold number or 'all'")
@click.option("--num-folds", default=10, type=int)
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]))
@click.option("--batch-size", default=128, type=int)
@click.option("--mask-mode", default="shuffle",
              type=click.Choice(["zero", "shuffle"]),
              help=("'zero' = vanilla joint zeroing (aggressive; non-RA members "
                    "collapse to majority class); 'shuffle' = "
                    "permutation-importance style cross-batch shuffle "
                    "(graceful, preserves marginals)."))
@click.option("--country", default=None, type=click.Choice([None, "JP", "TW"]),
              help="Stratify val by country (prefix match on filename)")
@click.option("--intensity", default=None, type=click.Choice([None, "L", "M", "H"]),
              help="Stratify val by DIEM-A intensity tag (filename suffix)")
@click.option("--output-dir", default=None,
              help="Where to save outputs (default docs/analysis/ensemble_part_masking/)")
@click.option("--members", default=None,
              help="Comma-separated experiment names (default = 7 PROD members)")
def main(fold: str, num_folds: int, device: str, batch_size: int,
         mask_mode: str, country: str | None, intensity: str | None,
         output_dir: str | None, members: str | None) -> None:
    env = EnvConfig()
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    click.echo(f"Device: {dev}")

    if members:
        members_tuple: tuple[str, ...] = tuple(m.strip() for m in members.split(","))
    else:
        members_tuple = PROD_MEMBERS

    if fold == "all":
        fold_ids = list(range(num_folds))
    else:
        fold_ids = [int(fold)]

    out_root = (Path(output_dir)
                if output_dir
                else env.project_root / "docs/analysis/ensemble_part_masking")
    out_root.mkdir(parents=True, exist_ok=True)
    click.echo(f"Output: {out_root}")
    click.echo(f"Mask mode: {mask_mode}")
    click.echo(f"Members ({len(members_tuple)}): {', '.join(members_tuple)}")

    all_rows: list[dict] = []
    aggregate_aucs: list[dict] = []
    for f_id in fold_ids:
        rec = _run_fold(
            fold_id=f_id, num_folds=num_folds,
            members=members_tuple, env=env, device=dev,
            batch_size=batch_size, country=country, intensity=intensity,
            out_root=out_root, mask_mode=mask_mode,
        )
        if rec is None:
            continue
        all_rows.extend(rec.pop("_csv_rows"))
        aggregate_aucs.append({
            "fold": f_id,
            "baseline_f1": rec["baseline_f1"],
            **{f"auc_{name}": rec["aucs"][name] for name in rec["aucs"]},
            "auc_gap_important_reverse": rec["aucs"]["important_first"]
            - rec["aucs"]["reverse"],
        })

    # ---- Aggregate CSV ----
    if all_rows:
        stratum_parts: list[str] = [mask_mode]
        if country:
            stratum_parts.append(country)
        if intensity:
            stratum_parts.append(f"int{intensity}")
        suffix = "_" + "_".join(stratum_parts)
        df = pd.DataFrame(all_rows)
        csv_path = out_root / f"ensemble_7way_part_importance{suffix}.csv"
        df.to_csv(csv_path, index=False)
        click.echo(f"\nSaved aggregated CSV: {csv_path}  ({len(df)} rows)")

        # Per-fold AUC summary
        auc_df = pd.DataFrame(aggregate_aucs)
        auc_path = out_root / f"ensemble_7way_aucs{suffix}.csv"
        auc_df.to_csv(auc_path, index=False)
        click.echo(f"Saved AUC summary: {auc_path}")

        # Cross-fold mean ± std for the headline gap
        if len(aggregate_aucs) > 0:
            gap = np.array([r["auc_gap_important_reverse"] for r in aggregate_aucs])
            click.echo(
                f"\nCross-fold headline: AUC gap (important − reverse) = "
                f"{gap.mean():+.4f} ± {gap.std(ddof=1) if len(gap) > 1 else 0.0:.4f} "
                f"(n={len(gap)} folds)"
            )

        # Heatmap (emotion × part F1 drop)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            agg = df.groupby("part").mean(numeric_only=True)
            drop_cols = [c for c in agg.columns if c.startswith("f1_drop_")]
            heat = agg[drop_cols].reindex(PARTS_ORDERED)
            heat.columns = [c.replace("f1_drop_", "") for c in heat.columns]
            fig, ax = plt.subplots(figsize=(8, 4))
            im = ax.imshow(heat.values, aspect="auto", cmap="RdYlBu_r")
            ax.set_xticks(range(len(heat.columns)))
            ax.set_xticklabels(heat.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(PARTS_ORDERED)))
            ax.set_yticklabels(PARTS_ORDERED)
            ax.set_title(
                "Ensemble (7-way PROD logit-mean): F1 drop per body part"
                + (f"  ({', '.join(stratum_parts)})" if stratum_parts else "")
            )
            fig.colorbar(im, ax=ax, label="F1 drop (baseline − masked)")
            fig.tight_layout()
            fig_path = out_root / f"ensemble_7way_part_heatmap{suffix}.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            click.echo(f"Saved heatmap: {fig_path}")
        except Exception as exc:
            click.echo(f"[warn] heatmap skipped: {exc}", err=True)


if __name__ == "__main__":
    main()
