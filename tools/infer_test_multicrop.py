""" exp068: multi-crop inference for the 7-way ensemble.

Re-evaluates OOF logits (validation set per fold) with multi-crop inference,
sweeping (policy, num_crops, aggregation), and optionally runs the same
multi-crop inference on the held-out test set to produce a candidate
submission CSV.

Design
------
1. For each ensemble member ``exp`` and fold ``k``:
   a. Load the val set indices from ``generate_lpo_splits``.
   b. For each clip in the val set:
      - Decode root_pos / joint_quats once (mirrors ``MotionDataset``).
      - Compute joint positions (FK) once for motion energy.
      - For each ``(policy, num_crops)`` combination, derive crop indices
        and build the model input ``(K, C, T, V)``.
   c. Run the model on the union of unique crops (deduplicated by
      ``(policy, num_crops)`` row). Save per-crop logits.
2. Aggregate per-crop logits across ``aggregation`` ∈ {logit_mean, prob_mean}
   and save per-setting OOF logits.
3. Compute ensemble F1 per setting and produce a results table.

CLI usage::

    python tools/infer_test_multicrop.py \
        --experiments exp020_conv1d_transformer_a01 \
        --num-folds 10 \
        --policies phase_shift,uniform_endpoints,center_quartiles,motion_energy_softmax \
        --num-crops 3,5,7 \
        --aggregations logit_mean,prob_mean \
        --output-dir experiments/exp068_multicrop_inference

See: §5.1 / §11.2.
Related: tools/multicrop_policies.py (policy primitives),
         tools/infer_test_ensemble.py (single-crop reference),
         tools/calibrate_ensemble.py (downstream consumer of OOF logits).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import click
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pybvh_ml  # noqa: E402

from utils.env import EnvConfig  # noqa: E402
from diema.data.splits import generate_lpo_splits  # noqa: E402
from diema.features.multi_stream import compute_joint_positions  # noqa: E402
from diema.models import build_model  # noqa: E402
import diema.models  # noqa: F401, E402  (register models)

from tools.multicrop_policies import (  # noqa: E402
    aggregate_logits,
    crop_indices_center_quartiles,
    crop_indices_motion_energy_softmax,
    crop_indices_phase_shift,
    crop_indices_uniform_endpoints,
    crop_starts_motion_energy_softmax,
    per_frame_motion_energy,
)
from tools.infer_test_ensemble import (  # noqa: E402
    _cfg_to_builder_ns,
    _load_checkpoint_into_model,
)


NUM_CLASSES = 12

# Default 7-way ensemble (matches tools/calibrate_ensemble.py).
DEFAULT_ENSEMBLE = [
    "exp002_a04_smooth",
    "exp003_ctrgcn_a00",
    "exp004_skateformer_a01",
    "exp006_protogcn_a00",
    "exp020_conv1d_transformer_a01",
    "exp023_keypoint_pool_mlp_a00",
    "exp034_regionaware_convtr_a00",
]

POLICY_NAMES = (
    "phase_shift",
    "uniform_endpoints",
    "center_quartiles",
    "motion_energy_softmax",
)


# ---------------------------------------------------------------------------
# Crop-index dispatcher
# ---------------------------------------------------------------------------

def get_crop_indices(
    policy: str,
    num_frames: int,
    clip_length: int,
    num_crops: int,
    energy: Optional[np.ndarray] = None,
    tau: float = 2.0,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (indices, weights) for the requested policy.

    indices: (num_crops, clip_length) integer array.
    weights: (num_crops,) for ``motion_energy_softmax``, ``None`` for others.
    """
    if policy == "phase_shift":
        return crop_indices_phase_shift(num_frames, clip_length, num_crops), None
    if policy == "uniform_endpoints":
        return crop_indices_uniform_endpoints(num_frames, clip_length, num_crops), None
    if policy == "center_quartiles":
        return crop_indices_center_quartiles(num_frames, clip_length, num_crops), None
    if policy == "motion_energy_softmax":
        if energy is None:
            raise ValueError("motion_energy_softmax requires `energy`")
        idx, w = crop_indices_motion_energy_softmax(
            num_frames, clip_length, num_crops, energy, tau=tau,
        )
        return idx, w
    raise ValueError(f"unknown policy '{policy}'")


# ---------------------------------------------------------------------------
# Model input construction (mirrors MotionDataset.__getitem__ steps 2-4)
# ---------------------------------------------------------------------------

def build_clip_data_ctv(
    clip: dict,
    target_repr: str,
    euler_orders=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one clip's quaternion data into the (C, T, V) tensor layout.

    Mirrors ``MotionDataset.__getitem__`` steps 2-3: quaternion → target_repr
    via ``pybvh_ml.convert_arrays`` (with ``euler_orders`` when needed), then
    pack to CTV via ``pybvh_ml.pack_to_ctv(..., center_root=False)``.

    For target_repr='euler', ``euler_orders`` MUST be provided
    (typically obtained from ``preprocessed['skeleton_info']['euler_orders']``);
    passing None raises a clear error to prevent silent miscalibration.

    Returns ``(data_ctv, joint_pos_world)`` where ``data_ctv`` has shape
    ``(C, num_frames, V)`` and ``joint_pos_world`` has shape
    ``(num_frames, J, 3)`` (used for motion energy).
    """
    root_pos = clip["root_pos"]
    joint_quats = clip["joint_data"]
    # Convert quaternion → target representation (default 6D).
    if target_repr != "quaternion":
        if target_repr == "euler" and euler_orders is None:
            raise ValueError(
                "target_repr='euler' requires euler_orders; pass it via "
                "preprocessed['skeleton_info']['euler_orders']"
            )
        stream = pybvh_ml.convert_arrays(
            joint_quats, "quaternion", target_repr, euler_orders=euler_orders,
        )
    else:
        stream = joint_quats
    data_ctv = pybvh_ml.pack_to_ctv(root_pos, stream, center_root=False)
    # FK for motion energy.
    joint_pos = compute_joint_positions(root_pos, joint_quats)
    return data_ctv.astype(np.float32), joint_pos.astype(np.float32)


def crop_ctv_to_input(
    data_ctv: np.ndarray, indices: np.ndarray,
) -> np.ndarray:
    """Apply ``indices`` (K, T) to ``data_ctv`` (C, total_frames, V).

    Returns ``(K, C, T, V)`` ready for batch inference.
    """
    K, T = indices.shape
    C, total_frames, V = data_ctv.shape
    # Modulo wrap-around for (rare) sequences shorter than clip_length.
    safe_idx = indices % max(total_frames, 1)
    out = data_ctv[:, safe_idx, :]  # (C, K, T, V) via fancy indexing
    out = out.transpose(1, 0, 2, 3)  # (K, C, T, V)
    return out.copy()  # contiguous


# ---------------------------------------------------------------------------
# Per-fold inference (one ensemble member, one fold)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference_for_fold(
    model: torch.nn.Module,
    val_indices: list[int],
    val_filenames: list[str],
    val_labels: np.ndarray,
    clips: list[dict],
    target_repr: str,
    clip_length: int,
    policies: tuple[str, ...],
    num_crops_list: tuple[int, ...],
    device: torch.device,
    batch_size: int = 64,
    motion_energy_tau: float = 2.0,
    euler_orders: Optional[list] = None,
) -> dict:
    """Run multi-crop inference on the val set of one (member, fold) pair.

    For each (policy, num_crops) configuration, we precompute the crop
    indices for every val clip and accumulate per-crop logits in a single
    pass through the dataloader. ``motion_energy_softmax`` weights are
    saved alongside.

    Returns a dict keyed by (policy, num_crops) → {
        "logits": (N_val, num_crops, NUM_CLASSES),
        "weights": (N_val, num_crops) float64 or None,
    }
    """
    model = model.to(device).eval()
    n_val = len(val_indices)

    # Precompute per-clip crops + weights, shared across all clips
    # of this fold (we need to know num_crops per setting to allocate output).
    # Output structure: per (policy, num_crops) → list of arrays, one per clip
    # of (num_crops, clip_length) indices.
    per_clip_configs: list[dict] = []  # one entry per val clip

    for clip_idx in val_indices:
        clip = clips[clip_idx]
        data_ctv, joint_pos = build_clip_data_ctv(
            clip, target_repr=target_repr,
            euler_orders=euler_orders,
        )
        num_frames = data_ctv.shape[1]
        # Compute motion energy once per clip (used for motion_energy_softmax).
        energy = per_frame_motion_energy(joint_pos)

        configs: dict[tuple[str, int], dict] = {}
        for policy in policies:
            for nc in num_crops_list:
                idx, weights = get_crop_indices(
                    policy, num_frames, clip_length, nc,
                    energy=energy, tau=motion_energy_tau,
                )
                configs[(policy, nc)] = {
                    "indices": idx,  # (nc, clip_length)
                    "weights": weights,  # (nc,) or None
                }
        per_clip_configs.append({
            "data_ctv": data_ctv,
            "configs": configs,
        })

    # Now: for each (policy, nc), run inference across all clips. We reuse
    # the data_ctv (decoding is expensive; do it once per clip).
    output: dict[tuple[str, int], dict] = {}
    for policy in policies:
        for nc in num_crops_list:
            all_per_crop_logits = np.zeros(
                (n_val, nc, NUM_CLASSES), dtype=np.float32,
            )
            all_weights = np.full((n_val, nc), 1.0 / nc, dtype=np.float64)

            # Build inputs in chunks for batched inference.
            buffer: list[tuple[int, np.ndarray]] = []  # (clip_pos, (nc, C, T, V))
            for clip_pos, entry in enumerate(per_clip_configs):
                cfg = entry["configs"][(policy, nc)]
                inp = crop_ctv_to_input(entry["data_ctv"], cfg["indices"])  # (nc, C, T, V)
                buffer.append((clip_pos, inp))
                if cfg["weights"] is not None:
                    all_weights[clip_pos] = cfg["weights"]

            # Flatten K crops × N clips into a single batch dim.
            stacked_inputs = np.concatenate(
                [inp for _, inp in buffer], axis=0,
            )  # (n_val * nc, C, T, V)

            n_total = stacked_inputs.shape[0]
            logits_flat = np.zeros((n_total, NUM_CLASSES), dtype=np.float32)
            for s in range(0, n_total, batch_size):
                e = min(s + batch_size, n_total)
                x = torch.from_numpy(stacked_inputs[s:e]).to(device)
                out = model(x)
                logits = out["logits"] if isinstance(out, dict) else out
                logits_flat[s:e] = logits.cpu().numpy()

            # Reshape back to (n_val, nc, NUM_CLASSES).
            all_per_crop_logits = logits_flat.reshape(n_val, nc, NUM_CLASSES)
            output[(policy, nc)] = {
                "logits": all_per_crop_logits,
                "weights": all_weights,
            }

    return {
        "configs": output,
        "filenames": list(val_filenames),
        "labels": val_labels,
    }


# ---------------------------------------------------------------------------
# Aggregation utilities
# ---------------------------------------------------------------------------

def aggregate_member_fold(
    per_crop_logits: np.ndarray,
    weights: np.ndarray,
    aggregation: str,
) -> np.ndarray:
    """Aggregate per-clip per-crop logits → per-clip logits.

    Fast-path: when all clips share identical per-crop weights (the common
    case for phase_shift / uniform_endpoints / center_quartiles policies,
    which use uniform 1/K weights), call ``aggregate_logits`` once with a
    1-D weight vector. Only ``motion_energy_softmax`` produces per-clip
    weights that vary across clips, requiring a per-clip loop.

    Args:
        per_crop_logits: (N, K, num_classes)
        weights: (N, K) — per-clip per-crop weights (sum to 1 along axis=1).
        aggregation: 'logit_mean' or 'prob_mean'.

    Returns:
        (N, num_classes) aggregated logits.
    """
    if per_crop_logits.shape[:2] != weights.shape:
        raise ValueError(
            f"shape mismatch: logits={per_crop_logits.shape}, weights={weights.shape}"
        )
    N, K, _ = per_crop_logits.shape
    # Fast path: all rows have identical weights → one vectorised call.
    if N > 0 and np.allclose(weights, weights[0:1], atol=1e-12, rtol=0):
        return aggregate_logits(per_crop_logits, aggregation, weights=weights[0])
    out = np.zeros((N, per_crop_logits.shape[2]), dtype=np.float64)
    for n in range(N):
        out[n] = aggregate_logits(
            per_crop_logits[n:n+1], aggregation, weights=weights[n],
        )[0]
    return out


# ---------------------------------------------------------------------------
# Macro-F1 helper (avoid sklearn re-import circle)
# ---------------------------------------------------------------------------

def macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(
        labels, preds,
        average="macro",
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    ))


# ---------------------------------------------------------------------------
# CLI driver — OOF mode
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--experiments", multiple=True, default=DEFAULT_ENSEMBLE,
    help="Ensemble member experiment folder names (default: production 7-way).",
)
@click.option("--num-folds", default=10, type=int)
@click.option("--folds", default="", type=str,
              help="Comma-separated fold ids to run (default: all 0..num_folds-1).")
@click.option(
    "--policies", default=",".join(POLICY_NAMES),
    help="Comma-separated subset of " + "|".join(POLICY_NAMES),
)
@click.option("--num-crops", default="3,5,7",
              help="Comma-separated K values to test (e.g. 3,5,7).")
@click.option("--aggregations", default="logit_mean,prob_mean",
              help="Comma-separated aggregations.")
@click.option("--motion-energy-tau", default=2.0, type=float,
              help="Softmax temperature for motion_energy_softmax policy.")
@click.option("--batch-size", default=64, type=int)
@click.option("--device", default="cuda",
              type=click.Choice(["cuda", "cpu"]))
@click.option("--output-dir", default="experiments/exp068_multicrop_inference",
              type=click.Path(file_okay=False))
@click.option("--save-per-crop/--no-save-per-crop", default=False,
              help="Save per-crop logits for downstream re-aggregation.")
@click.option("--quick/--full", default=False,
              help="Quick mode: 1 ensemble member, 1 fold (smoke test).")
def main(
    experiments: tuple,
    num_folds: int,
    folds: str,
    policies: str,
    num_crops: str,
    aggregations: str,
    motion_energy_tau: float,
    batch_size: int,
    device: str,
    output_dir: str,
    save_per_crop: bool,
    quick: bool,
) -> None:
    env = EnvConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if quick:
        if experiments and len(experiments) > 1:
            click.echo(
                f"[quick] keeping only first experiment: {experiments[0]} "
                f"(dropping {experiments[1:]})",
                err=True,
            )
        experiments = (experiments[0],) if experiments else (DEFAULT_ENSEMBLE[0],)
        folds = folds or "0"

    if not folds.strip():
        fold_ids = list(range(num_folds))
    else:
        try:
            fold_ids = [int(x) for x in folds.split(",")]
        except ValueError as exc:
            raise click.UsageError(f"--folds parsing failed: {exc}") from exc
        for f in fold_ids:
            if not (0 <= f < num_folds):
                raise click.UsageError(
                    f"--folds entry {f} out of range [0, {num_folds})"
                )
    if not fold_ids:
        raise click.UsageError("--folds must specify at least one fold id")

    policy_list = tuple(p.strip() for p in policies.split(",") if p.strip())
    for p in policy_list:
        if p not in POLICY_NAMES:
            raise click.UsageError(f"unknown policy '{p}'")
    if not policy_list:
        raise click.UsageError("--policies must list at least one policy")

    try:
        nc_list_raw = [int(x) for x in num_crops.split(",") if x.strip()]
    except ValueError as exc:
        raise click.UsageError(f"--num-crops parsing failed: {exc}") from exc
    if not nc_list_raw:
        raise click.UsageError("--num-crops must list at least one K value")
    for nc in nc_list_raw:
        if nc < 1:
            raise click.UsageError(f"--num-crops entries must be >= 1, got {nc}")
    # Deduplicate K values (sorted) so that "3,3,5" doesn't run K=3 twice.
    nc_list = tuple(sorted(set(nc_list_raw)))

    agg_list = tuple(a.strip() for a in aggregations.split(",") if a.strip())
    for a in agg_list:
        if a not in ("logit_mean", "prob_mean"):
            raise click.UsageError(f"unknown aggregation '{a}'")
    if not agg_list:
        raise click.UsageError("--aggregations must list at least one aggregation")
    if motion_energy_tau <= 0:
        raise click.UsageError("--motion-energy-tau must be > 0")

    dev = torch.device(device if (torch.cuda.is_available() or device == "cpu") else "cpu")
    click.echo(f"Inference device: {dev}")
    click.echo(f"Experiments: {experiments}")
    click.echo(f"Folds: {fold_ids}")
    click.echo(f"Policies: {policy_list}")
    click.echo(f"num_crops: {nc_list}")
    click.echo(f"aggregations: {agg_list}")

    # Load training data once (filenames/clips are shared across all members).
    data_path = env.processed_dir / "motion_train_quat.npz"
    preprocessed = pybvh_ml.load_preprocessed(str(data_path))
    all_filenames = preprocessed["filenames"]
    all_clips = preprocessed["clips"]
    all_labels = preprocessed["labels"]
    skeleton_info = preprocessed.get("skeleton_info", {}) or {}
    dataset_euler_orders = (
        skeleton_info.get("euler_orders") if isinstance(skeleton_info, dict) else None
    )
    splits = generate_lpo_splits(all_filenames, num_folds=num_folds)

    # ------------------------------------------------------------------
    # For each (member exp, fold), run multi-crop inference and dump
    # per-(policy, num_crops, agg) aggregated logits.
    # ------------------------------------------------------------------
    # We accumulate results in a nested dict for ensemble averaging later.
    # Structure: results[exp][fold][(policy, nc, agg)] = (logits_NxC, labels_N, fnames_N)

    per_member_per_fold: dict[str, dict[int, dict]] = {}
    for exp in experiments:
        per_member_per_fold[exp] = {}
        exp_dir = env.artifacts_dir / exp
        for fold in fold_ids:
            ckpt_path = exp_dir / f"fold_{fold:02d}" / "best.ckpt"
            cfg_path = exp_dir / f"fold_{fold:02d}" / "config.yaml"
            if not ckpt_path.exists():
                click.echo(f"[skip] {exp}/fold_{fold:02d}: ckpt missing", err=True)
                continue

            cfg_dict = OmegaConf.to_container(OmegaConf.load(str(cfg_path)), resolve=True)
            target_repr = cfg_dict.get("target_repr", "6d")
            clip_length = cfg_dict.get("clip_length", 64)

            # Build model and load weights
            builder_ns = _cfg_to_builder_ns(cfg_dict)
            model = build_model(builder_ns)
            _load_checkpoint_into_model(ckpt_path, model)

            split = splits[fold]
            val_pairs = split["val"]
            val_indices = [idx for _, idx in val_pairs]
            val_filenames = [fn for fn, _ in val_pairs]
            val_labels = np.asarray(
                [int(all_labels[i]) for i in val_indices], dtype=np.int64,
            )

            click.echo(
                f"\n=== {exp} fold_{fold:02d}: "
                f"{len(val_indices)} val samples, target_repr={target_repr}, clip={clip_length}"
            )

            run_out = run_inference_for_fold(
                model=model,
                val_indices=val_indices,
                val_filenames=val_filenames,
                val_labels=val_labels,
                clips=all_clips,
                target_repr=target_repr,
                clip_length=clip_length,
                policies=policy_list,
                num_crops_list=nc_list,
                device=dev,
                batch_size=batch_size,
                motion_energy_tau=motion_energy_tau,
                euler_orders=dataset_euler_orders,
            )

            per_member_per_fold[exp][fold] = run_out

            # Optionally save per-crop logits (heavy: K * N_val * NUM_CLASSES per setting).
            if save_per_crop:
                save_dir = out / "per_crop_logits" / exp / f"fold_{fold:02d}"
                save_dir.mkdir(parents=True, exist_ok=True)
                for (policy, nc), conf in run_out["configs"].items():
                    np.savez_compressed(
                        save_dir / f"{policy}_K{nc}.npz",
                        logits=conf["logits"],
                        weights=conf["weights"],
                        labels=val_labels,
                    )

            # Free GPU memory before next ckpt.
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Aggregate per setting:
    #   For each (policy, nc, agg): compute per-fold logits per member,
    #   ensemble across members (equal-weight), then concatenate folds.
    # ------------------------------------------------------------------
    all_settings = []
    for policy in policy_list:
        for nc in nc_list:
            for agg in agg_list:
                all_settings.append((policy, nc, agg))

    # First pass: produce per-(member, fold, setting) aggregated logits.
    per_member_per_fold_aggregated: dict[str, dict[int, dict]] = {}
    for exp, fold_map in per_member_per_fold.items():
        per_member_per_fold_aggregated[exp] = {}
        for fold, run_out in fold_map.items():
            per_member_per_fold_aggregated[exp][fold] = {}
            for (policy, nc), conf in run_out["configs"].items():
                logits_K = conf["logits"]   # (N, nc, num_classes)
                weights = conf["weights"]   # (N, nc)
                for agg in agg_list:
                    agg_logits = aggregate_member_fold(logits_K, weights, agg)
                    per_member_per_fold_aggregated[exp][fold][(policy, nc, agg)] = {
                        "logits": agg_logits,
                        "labels": run_out["labels"],
                        "filenames": run_out["filenames"],
                    }

    # Second pass: ensemble across members (equal-weight) per fold.
    setting_results: dict[tuple, dict] = {}
    for setting in all_settings:
        per_fold_agg_logits = []
        per_fold_labels = []
        per_fold_filenames = []
        for fold in fold_ids:
            members_in_fold = [
                per_member_per_fold_aggregated[exp][fold][setting]
                for exp in experiments
                if fold in per_member_per_fold_aggregated.get(exp, {})
            ]
            if not members_in_fold:
                continue
            stacked = np.stack(
                [m["logits"] for m in members_in_fold], axis=0,
            )  # (M, N, num_classes)
            mean_logits = stacked.mean(axis=0)  # (N, num_classes)
            per_fold_agg_logits.append(mean_logits)
            per_fold_labels.append(members_in_fold[0]["labels"])
            per_fold_filenames.append(members_in_fold[0]["filenames"])

        if not per_fold_agg_logits:
            continue
        all_logits = np.concatenate(per_fold_agg_logits, axis=0)
        all_labels_arr = np.concatenate(per_fold_labels, axis=0)
        all_fnames = [fn for chunk in per_fold_filenames for fn in chunk]
        preds = all_logits.argmax(axis=1)
        per_fold_f1 = [
            macro_f1(L.argmax(axis=1), Y)
            for L, Y in zip(per_fold_agg_logits, per_fold_labels)
        ]
        setting_results[setting] = {
            "policy": setting[0],
            "num_crops": int(setting[1]),
            "aggregation": setting[2],
            "overall_f1": macro_f1(preds, all_labels_arr),
            "overall_acc": float((preds == all_labels_arr).mean()),
            "per_fold_f1": list(map(float, per_fold_f1)),
            "per_fold_f1_mean": float(np.mean(per_fold_f1)),
            "per_fold_f1_std": float(np.std(per_fold_f1, ddof=1)) if len(per_fold_f1) > 1 else 0.0,
            "n_folds_present": len(per_fold_f1),
            "n_samples": int(all_logits.shape[0]),
        }

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    summary = {
        "config": {
            "experiments": list(experiments),
            "fold_ids": fold_ids,
            "policies": list(policy_list),
            "num_crops": list(nc_list),
            "aggregations": list(agg_list),
            "motion_energy_tau": float(motion_energy_tau),
            "quick_mode": bool(quick),
        },
        "settings": [
            {**v, "policy": k[0], "num_crops": int(k[1]), "aggregation": k[2]}
            for k, v in setting_results.items()
        ],
    }
    json_path = out / "results.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    click.echo(f"\nWrote {json_path}")

    # Print sorted summary
    rows = sorted(
        summary["settings"],
        key=lambda r: r["overall_f1"], reverse=True,
    )
    click.echo("\n=== Settings sorted by overall_f1 ===")
    click.echo(f"{'rank':>4}  {'policy':<24}  {'K':>3}  {'agg':<11}  {'F1':>7}  {'std':>6}  {'#folds':>6}")
    for i, r in enumerate(rows[:30], start=1):
        click.echo(
            f"{i:>4}  {r['policy']:<24}  {r['num_crops']:>3}  {r['aggregation']:<11}  "
            f"{r['overall_f1']*100:>6.3f}%  {r['per_fold_f1_std']*100:>5.3f}  {r['n_folds_present']:>6}"
        )


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


if __name__ == "__main__":
    main()
