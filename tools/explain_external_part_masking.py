""" ensemble part-masking for skeleton-based external members.

Related:
- tools/explain_ensemble_part_masking.py
    Sibling tool for the 7 PROD members. Same DIEMA 6-part partition,
    same mask modes (zero / shuffle), same per-fold val protocol — but
    masking is applied in the *projected* joint space (H36M-17 for
    MotionBERT, NTU-25 for MAMP) before the frozen encoder.
- tools/extract_motionbert_features_for_diema.py
- tools/extract_mamp_features_for_diema.py
    Source of the build_*/load_*/feature_extraction routines we reuse.
- diema/features/bvh_to_h36m17.py
- diema/features/bvh_to_ntu25.py
    Projection pipelines.

Outputs (per external member, per fold, per mask key) the val-set logits
used by the 11-way fusion in `tools/fuse_ensemble_masked.py` (which
combines these with the PROD logits produced by
`tools/explain_ensemble_part_masking.py`).

Mask is applied at the *projected* joint level. The DIEMA → projected
mapping is documented in
``docs/results_summary.md`` §18.6 and inlined as constants below.

The frozen encoder runs un-masked at training time (using cached
features from the standard extract_* scripts). At masking-evaluation
time we re-extract features ONLY on the val fold with the joints zeroed
or shuffled, then predict via the per-fold sklearn probe trained on the
cached un-masked train features.

Usage::

    # MotionBERT
    python tools/explain_external_part_masking.py \
        --member motionbert --fold all --mask-mode shuffle

    # MAMP NTU60 X-subject
    python tools/explain_external_part_masking.py \
        --member mamp60 --fold all --mask-mode shuffle

    # MAMP NTU120 X-set
    python tools/explain_external_part_masking.py \
        --member mamp120 --fold all --mask-mode shuffle
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pybvh_ml  # noqa: F401,E402
from utils.env import EnvConfig  # noqa: E402
from diema.data.parser import emotion_to_label, IDX_TO_EMOTION  # noqa: E402
from diema.data.splits import generate_lpo_splits  # noqa: E402
from diema.models.skeleton_graph import DIEMA_BODY_PARTS  # noqa: E402
from diema.features.bvh_to_h36m17 import project_bvh_to_h36m17  # noqa: E402
from diema.features.bvh_to_ntu25 import project_bvh_to_ntu25  # noqa: E402

from tools.explain_ensemble_part_masking import (  # noqa: E402
    PARTS_ORDERED,
    _mask_key,
    _orderings_for_drops,
    _resolve_mask_keys_for_curves,
    _faithfulness_curve_from_cache,
    _logit_mean_fuse,
    _logit_mean_fuse_argmax,
    _ensemble_macro_f1,
)
from tools.explain_part_masking import _macro_f1, _per_class_f1, _auc_drop  # noqa: E402


NUM_CLASS = 12


# --------------------------------------------------------------------------- #
# DIEMA 6-part → projected joint indices (matches docs/results_summary.md §18.6)
# --------------------------------------------------------------------------- #


# H36M-17 (MotionBERT). Verified against
# diema/features/bvh_to_h36m17.py H36M_TO_BVH semantics + the H36M order
# in tmp/MotionBERT/lib/data/dataset_action.py.
H36M17_PART_INDICES: dict[str, tuple[int, ...]] = {
    "torso": (0, 7),               # root, belly
    "head":  (8, 9, 10),           # neck, nose, head
    "r_arm": (14, 15, 16),         # rsho, relb, rwri
    "l_arm": (11, 12, 13),         # lsho, lelb, lwri
    "r_leg": (1, 2, 3),            # rhip, rkne, rank
    "l_leg": (4, 5, 6),            # lhip, lkne, lank
}
assert sorted(set().union(*H36M17_PART_INDICES.values())) == list(range(17)), (
    "H36M-17 part mapping must partition 0..16"
)


# NTU-25 (MAMP). Verified against diema/features/bvh_to_ntu25.py
# NTU25_TO_BVH semantics. Hand-tips / thumbs duplicate the wrist/hand
# joints in BVH-24 (no finger detail in source skeleton).
NTU25_PART_INDICES: dict[str, tuple[int, ...]] = {
    "torso": (0, 1, 20),           # SpineBase, SpineMid, SpineShoulder
    "head":  (2, 3),               # Neck, Head
    "r_arm": (8, 9, 10, 11, 23, 24),   # RSho, REl, RWri, RHand, RHandTip, RThumb
    "l_arm": (4, 5, 6, 7, 21, 22),     # LSho, LEl, LWri, LHand, LHandTip, LThumb
    "r_leg": (16, 17, 18, 19),     # RHip, RKnee, RAnkle, RFoot
    "l_leg": (12, 13, 14, 15),     # LHip, LKnee, LAnkle, LFoot
}
assert sorted(set().union(*NTU25_PART_INDICES.values())) == list(range(25)), (
    "NTU-25 part mapping must partition 0..24"
)


def _proj_part_indices(member: str) -> dict[str, tuple[int, ...]]:
    if member == "motionbert":
        return H36M17_PART_INDICES
    if member in ("mamp60", "mamp120"):
        return NTU25_PART_INDICES
    raise ValueError(f"unknown member: {member!r}")


# --------------------------------------------------------------------------- #
# Member-specific encoder / projection
# --------------------------------------------------------------------------- #


def _import_motionbert():
    """Import MotionBERT-side helpers (no module-level torch state)."""
    from tools.extract_motionbert_features_for_diema import (
        build_motionbert_lite, load_pretrained,
    )
    return build_motionbert_lite, load_pretrained


def _import_mamp():
    """Import MAMP-side helpers (build, load, frozen-encoder forward)."""
    from tools.extract_mamp_features_for_diema import (
        build_mamp_encoder, load_pretrained, forward_features,
    )
    return build_mamp_encoder, load_pretrained, forward_features


def _project_clip(member: str, root_pos: np.ndarray, joint_quats: np.ndarray,
                  target_frames: int, normalize: str | None) -> np.ndarray:
    """Run the BVH → projected-joint projection used by ``member``.

    Returns ``(T, V, 3)`` float32 — un-masked.
    """
    if member == "motionbert":
        # bvh_to_h36m17 returns (T, 17, 3) — note: the MotionBERT extraction
        # script keeps the projected layout as 17 joints; orthographic XZ
        # projection happens inside the projector.
        return project_bvh_to_h36m17(root_pos, joint_quats, target_frames=target_frames)
    if member in ("mamp60", "mamp120"):
        return project_bvh_to_ntu25(
            root_pos, joint_quats, target_frames=target_frames,
            normalize=normalize or "unit",
        )
    raise ValueError(f"unknown member: {member!r}")


def _extract_features_masked(
    member: str,
    model: torch.nn.Module,
    val_clips: list[tuple[np.ndarray, np.ndarray]],  # (root_pos, joint_quats)
    mask_node_indices: tuple[int, ...] | None,
    perm_seed: int | None,
    device: torch.device,
    target_frames: int,
    normalize: str | None,
    batch_size: int,
) -> np.ndarray:
    """Project + (optionally) mask val clips + run the frozen encoder.

    Returns the per-joint features array used by the existing sklearn
    probe (matches the shape of ``features_per_joint`` in the cached
    NPZ): ``(N_val, V_proj, dim)`` float32.

    ``mask_node_indices`` are *projected*-space joint indices (H36M-17
    or NTU-25, depending on member). ``perm_seed=None`` and
    ``mask_node_indices=None`` together encode "baseline / no mask".

    When ``perm_seed`` is given, we apply permutation-importance shuffle
    on the projected joints (matches `_apply_mask_inplace(mode='shuffle')`
    semantics from the PROD tool).
    """
    V_proj = 17 if member == "motionbert" else 25
    N = len(val_clips)

    # Project all clips up front so we can apply a deterministic per-batch
    # shuffle without worrying about variable BVH lengths.
    projected = np.zeros((N, target_frames, V_proj, 3), dtype=np.float32)
    for i, (rp, jq) in enumerate(val_clips):
        projected[i] = _project_clip(
            member, rp, jq, target_frames=target_frames, normalize=normalize,
        )

    # Mask the projected joints (zero or shuffle).
    if mask_node_indices is not None and len(mask_node_indices) > 0:
        if perm_seed is None:
            # Zero mask
            for n in mask_node_indices:
                projected[:, :, n, :] = 0.0
        else:
            # Shuffle mask — cross-sample permutation on the V axis slice.
            rng = np.random.default_rng(perm_seed)
            perm = rng.permutation(N)
            for n in mask_node_indices:
                projected[:, :, n, :] = projected[perm, :, n, :]

    # Run the frozen encoder in batches.
    if member == "motionbert":
        feats = np.zeros((N, V_proj, 512), dtype=np.float32)
    else:  # mamp60 / mamp120
        feats = np.zeros((N, V_proj, 256), dtype=np.float32)

    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        if member == "motionbert":
            x = torch.from_numpy(projected[s:e]).to(device)
            with torch.no_grad():
                rep = model(x, return_rep=True)  # (B, T, 17, 512)
            feats[s:e] = rep.mean(axis=1).cpu().numpy()
        else:
            # MAMP expects (B, C=3, T, V=25, M=1)
            x = torch.from_numpy(projected[s:e]).permute(0, 3, 1, 2).unsqueeze(-1).to(device)
            from tools.extract_mamp_features_for_diema import forward_features
            with torch.no_grad():
                feat = forward_features(model, x)  # (B, M, TP, VP, D)
            # Aggregate matches extract script: mean over (M, TP) → (B, V_proj, D)
            feats[s:e] = feat.mean(dim=(1, 2)).cpu().numpy()
    return feats


# --------------------------------------------------------------------------- #
# Per-member I/O config
# --------------------------------------------------------------------------- #


MEMBER_CONFIG: dict[str, dict] = {
    "motionbert": {
        "features_npz": "data/diema_challenge/processed/motionbert_features_b00_train.npz",
        "checkpoint":   "data/motionbert/checkpoints/MB_pretrain_lite.bin",
        "target_frames": 243,
        "normalize": None,
        "batch_size": 32,
        # Existing OOF cache (sanity-check baseline match):
        "oof_logits_npy":
            "output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_logits.npy",
        "oof_filenames_npy":
            "output/predictions/motionbert_lite_per_joint/exp081_motionbert_oof_filenames.npy",
        "probe_C": 1.0,
    },
    "mamp60": {
        "features_npz": "data/diema_challenge/processed/mamp_ntu60xsub_train.npz",
        "checkpoint":   "data/mamp/checkpoints/ntu60_xsub.pth",
        "target_frames": 120,
        "normalize": "unit",
        "batch_size": 16,
        "oof_logits_npy":  "output/predictions/mamp_ntu60xsub/oof_logits.npy",
        "oof_filenames_npy": "output/predictions/mamp_ntu60xsub/oof_filenames.npy",
        "probe_C": 1.0,
    },
    "mamp120": {
        "features_npz": "data/diema_challenge/processed/mamp_ntu120xset_train.npz",
        "checkpoint":   "data/mamp/checkpoints/ntu120_xset.pth",
        "target_frames": 120,
        "normalize": "unit",
        "batch_size": 16,
        "oof_logits_npy":  "output/predictions/mamp_ntu120xset/oof_logits.npy",
        "oof_filenames_npy": "output/predictions/mamp_ntu120xset/oof_filenames.npy",
        "probe_C": 1.0,
    },
}


def _build_member_model(member: str, device: torch.device) -> torch.nn.Module:
    cfg = MEMBER_CONFIG[member]
    if member == "motionbert":
        build, load = _import_motionbert()
        return load(build(), cfg["checkpoint"], device)
    else:
        build, load, _fwd = _import_mamp()
        return load(build(), cfg["checkpoint"], device)


# --------------------------------------------------------------------------- #
# Per-fold driver
# --------------------------------------------------------------------------- #


def _run_member_fold(
    member: str,
    fold_id: int,
    num_folds: int,
    mask_mode: str,
    env: EnvConfig,
    device: torch.device,
    out_root: Path,
) -> dict | None:
    """Run baseline + 6 single-part + cumulative masks for one fold."""
    cfg = MEMBER_CONFIG[member]

    # ---- 1. Load cached un-masked features for probe training ----
    feats_npz = np.load(cfg["features_npz"], allow_pickle=True)
    cached_filenames = np.asarray(feats_npz["filenames"])
    per_joint = np.asarray(feats_npz["features_per_joint"], dtype=np.float32)
    cached_labels = np.array(
        [emotion_to_label(str(fn)) for fn in cached_filenames], dtype=np.int64
    )

    # ---- 2. LPO split on the same filename array ----
    splits = generate_lpo_splits(cached_filenames.tolist(), num_folds=num_folds)
    train_indices = [idx for _, idx in splits[fold_id]["train"]]
    val_indices = [idx for _, idx in splits[fold_id]["val"]]

    # Train probe on un-masked train features (flattened per_joint).
    X_tr = per_joint[train_indices].reshape(len(train_indices), -1)
    y_tr = cached_labels[train_indices]
    y_va = cached_labels[val_indices]
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(
        C=cfg["probe_C"], max_iter=2000, n_jobs=-1, solver="lbfgs", random_state=42,
    )
    clf.fit(scaler.transform(X_tr), y_tr)

    # ---- 3. Load val BVH clips for projection + masked extraction ----
    # The extract_*_features_for_diema.py scripts use np.load directly (not
    # pybvh_ml.load_preprocessed), so the keys are the flat
    # ``clip_{i}_root_pos`` / ``clip_{i}_joint_data`` style. We follow the
    # same convention for parity with the cached feature extraction.
    raw_npz_path = env.processed_dir / "motion_train_quat.npz"
    raw = np.load(str(raw_npz_path), allow_pickle=True)
    raw_filenames = np.asarray(raw["filenames"])
    raw_index = {str(fn): i for i, fn in enumerate(raw_filenames)}
    val_clips: list[tuple[np.ndarray, np.ndarray]] = []
    for idx in val_indices:
        fn = str(cached_filenames[idx])
        if fn not in raw_index:
            raise RuntimeError(
                f"val filename {fn!r} not found in raw NPZ at {raw_npz_path}"
            )
        ri = raw_index[fn]
        rp = np.asarray(raw[f"clip_{ri}_root_pos"], dtype=np.float32)
        jq = np.asarray(raw[f"clip_{ri}_joint_data"], dtype=np.float32)
        val_clips.append((rp, jq))

    # ---- 4. Load frozen encoder ----
    model = _build_member_model(member, device)

    # ---- 5. Single-part + baseline mask keys ----
    part_indices = _proj_part_indices(member)
    single_keys: list[tuple[str, ...]] = [()] + [(p,) for p in PARTS_ORDERED]
    fused_val_logits_by_key: dict[tuple[str, ...], np.ndarray] = {}

    for key in single_keys:
        mask_nodes: tuple[int, ...] | None
        if len(key) == 0:
            mask_nodes = None
        else:
            mask_nodes = tuple(sorted({
                i for p in key for i in part_indices[p]
            }))
        perm_seed = None
        if mask_mode == "shuffle" and mask_nodes is not None:
            perm_seed = fold_id * 1000
        t0 = time.time()
        feats = _extract_features_masked(
            member, model, val_clips, mask_nodes, perm_seed, device,
            target_frames=cfg["target_frames"], normalize=cfg["normalize"],
            batch_size=cfg["batch_size"],
        )
        X_va = feats.reshape(feats.shape[0], -1)
        proba = clf.predict_proba(scaler.transform(X_va))
        # Project sklearn's class set into the full 12-class log-prob space.
        logits = np.full((len(X_va), NUM_CLASS), -np.inf, dtype=np.float32)
        for j, cls in enumerate(clf.classes_):
            logits[:, int(cls)] = np.log(np.clip(proba[:, j], 1e-12, 1.0)).astype(np.float32)
        # Replace -inf placeholders for unseen classes with a very small log-prob
        # so the downstream logit-mean isn't dominated by a -inf member.
        logits = np.where(np.isfinite(logits), logits, -50.0)
        fused_val_logits_by_key[key] = logits
        key_label = "baseline" if len(key) == 0 else "+".join(key)
        click.echo(
            f"  fold_{fold_id:02d} {member} mask {key_label:<20} "
            f"F1={_macro_f1(y_va, logits.argmax(1))*100:.2f}%  ({time.time()-t0:.1f}s)"
        )

    # ---- 6. Cumulative masks ----
    base_f1 = _macro_f1(y_va, fused_val_logits_by_key[()].argmax(1))
    part_f1 = {
        p: _macro_f1(y_va, fused_val_logits_by_key[(p,)].argmax(1))
        for p in PARTS_ORDERED
    }
    drops = {p: base_f1 - part_f1[p] for p in PARTS_ORDERED}
    cumulative_keys, important, reverse, random_order = (
        _resolve_mask_keys_for_curves(drops, seed=fold_id)
    )
    needed = [k for k in cumulative_keys if k not in fused_val_logits_by_key]
    for key in needed:
        mask_nodes = tuple(sorted({i for p in key for i in part_indices[p]}))
        perm_seed = (fold_id * 1000) if mask_mode == "shuffle" else None
        feats = _extract_features_masked(
            member, model, val_clips, mask_nodes, perm_seed, device,
            target_frames=cfg["target_frames"], normalize=cfg["normalize"],
            batch_size=cfg["batch_size"],
        )
        X_va = feats.reshape(feats.shape[0], -1)
        proba = clf.predict_proba(scaler.transform(X_va))
        logits = np.full((len(X_va), NUM_CLASS), -50.0, dtype=np.float32)
        for j, cls in enumerate(clf.classes_):
            logits[:, int(cls)] = np.log(np.clip(proba[:, j], 1e-12, 1.0)).astype(np.float32)
        fused_val_logits_by_key[key] = logits

    # ---- 7. Save per-fold per-mask logits (one .npy per key) ----
    save_dir = out_root / "member_masked_logits" / member / f"fold_{fold_id:02d}_{mask_mode}"
    save_dir.mkdir(parents=True, exist_ok=True)
    for key, lg in fused_val_logits_by_key.items():
        key_name = "baseline" if len(key) == 0 else "_".join(sorted(key))
        np.save(save_dir / f"{key_name}.npy", lg.astype(np.float32))
    np.save(save_dir / "_labels.npy", y_va.astype(np.int64))
    # Save the filename order so external-side data lines up with PROD cache.
    np.save(save_dir / "_filenames.npy",
            np.asarray([str(cached_filenames[i]) for i in val_indices]))

    # Per-fold record summary for index.
    record = {
        "member": member,
        "fold": fold_id,
        "mask_mode": mask_mode,
        "n_val": int(len(y_va)),
        "baseline_f1": float(base_f1),
        "part_f1": {p: float(v) for p, v in part_f1.items()},
        "drops_by_part": {p: float(v) for p, v in drops.items()},
    }
    return record


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@click.command()
@click.option("--member", required=True,
              type=click.Choice(["motionbert", "mamp60", "mamp120"]))
@click.option("--fold", default="0", help="Fold number or 'all'")
@click.option("--num-folds", default=10, type=int)
@click.option("--mask-mode", default="shuffle",
              type=click.Choice(["zero", "shuffle"]))
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]))
@click.option("--output-dir", default=None,
              help="Defaults to docs/analysis/ensemble_part_masking/")
def main(member: str, fold: str, num_folds: int, mask_mode: str,
         device: str, output_dir: str | None) -> None:
    env = EnvConfig()
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    out_root = (Path(output_dir) if output_dir
                else env.project_root / "docs/analysis/ensemble_part_masking")
    out_root.mkdir(parents=True, exist_ok=True)

    if fold == "all":
        fold_ids = list(range(num_folds))
    else:
        fold_ids = [int(fold)]

    records: list[dict] = []
    for f_id in fold_ids:
        click.echo(f"=== {member} fold_{f_id:02d} ({mask_mode}) ===")
        rec = _run_member_fold(
            member=member, fold_id=f_id, num_folds=num_folds,
            mask_mode=mask_mode, env=env, device=dev, out_root=out_root,
        )
        if rec is not None:
            records.append(rec)

    # Aggregate summary across folds for this member.
    import json
    summary = {
        "member": member,
        "mask_mode": mask_mode,
        "n_folds": len(records),
        "baseline_f1_mean": float(np.mean([r["baseline_f1"] for r in records])),
        "baseline_f1_std": float(np.std([r["baseline_f1"] for r in records], ddof=1))
            if len(records) > 1 else 0.0,
        "drops_by_part_mean": {
            p: float(np.mean([r["drops_by_part"][p] for r in records]))
            for p in PARTS_ORDERED
        },
        "per_fold": records,
    }
    out_path = out_root / f"{member}_{mask_mode}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    click.echo(f"\nSaved {out_path}")
    click.echo(f"Baseline F1: {summary['baseline_f1_mean']*100:.2f} ± "
               f"{summary['baseline_f1_std']*100:.2f} %")
    click.echo(f"Per-part drops (mean across folds):")
    for p in PARTS_ORDERED:
        click.echo(f"  {p:>6}: {summary['drops_by_part_mean'][p]*100:+.3f} pp")


if __name__ == "__main__":
    main()
