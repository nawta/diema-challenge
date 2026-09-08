""" ensemble part-masking for the C3D contact-only member.

Related:
- tools/explain_ensemble_part_masking.py    (PROD side)
- tools/explain_external_part_masking.py    (skeleton-based externals)
- experiments/exp072_c3d_contact_only/run_smoke.py   (probe config + 22-D contact subset)
- diema/data/c3d_parser.py                  (read_c3d, VICON_PRODUCTION_PART_GROUPS)
- diema/features/c3d_stats.py               (compute_c3d_stats, feature_names)
- diema/data/splits.generate_lpo_splits     (LPO 10-fold splits)

C3D inputs are 57 Vicon Production markers, not skeleton joints. We map
DIEMA 6 parts to Vicon marker groups via the existing
``VICON_PRODUCTION_PART_GROUPS`` (which has 7 part labels: head,
torso_spine, l_arm, r_arm, pelvis, l_leg, r_leg). DIEMA "torso" subsumes
``torso_spine + pelvis``; the other 5 names are 1-to-1.

For each fold and each mask, we:
  1. Load raw .c3d files for the val performers (cached per fold).
  2. Apply marker-level mask (zero / cross-sample shuffle on the val set).
  3. Recompute the 51-D ``compute_c3d_stats`` per masked clip, then
     subset to the 22-D contact-only feature set (matches exp072).
  4. Train sklearn LogisticRegression(C=1.0) on the un-masked train
     22-D features (cached in ``output/c3d_stats_cache.npz``).
  5. Predict on the masked val 22-D features → val logits.

Saves to ``docs/analysis/ensemble_part_masking/member_masked_logits/c3d/fold_NN_<mode>/``
in the same ``{mask_key}.npy`` format the fusion script consumes.

Usage::

    python tools/explain_c3d_part_masking.py --fold all --mask-mode shuffle
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.env import EnvConfig  # noqa: E402
from diema.data.parser import emotion_to_label  # noqa: E402
from diema.data.splits import generate_lpo_splits  # noqa: E402
from diema.data.c3d_parser import (  # noqa: E402
    read_c3d,
    group_markers_by_part,
    VICON_PRODUCTION_PART_GROUPS,
)
from diema.features.c3d_stats import (  # noqa: E402
    compute_c3d_stats,
    stats_to_vector,
)

from tools.explain_ensemble_part_masking import (  # noqa: E402
    PARTS_ORDERED,
    _resolve_mask_keys_for_curves,
)
from tools.explain_part_masking import _macro_f1  # noqa: E402


NUM_CLASS = 12


# DIEMA 6-part → C3D Vicon part-group names.
# C3D groups have 7 names; DIEMA "torso" subsumes ``torso_spine + pelvis``.
DIEMA_TO_C3D_GROUPS: dict[str, tuple[str, ...]] = {
    "torso": ("torso_spine", "pelvis"),
    "head":  ("head",),
    "l_arm": ("l_arm",),
    "r_arm": ("r_arm",),
    "l_leg": ("l_leg",),
    "r_leg": ("r_leg",),
}


# Contact-only 22-D subset (matches exp072_c3d_contact_only/run_smoke.py).
CONTACT_FEATURES_22D: list[int] = [
    2, 3, 4, 5, 6,            # global_mean_speed, max_speed, mean_accel, max_accel, mean_jerk
    9, 10, 13, 14, 17, 18,    # head, torso, l_arm speeds
    21, 22, 25, 26, 29, 30,   # r_arm, pelvis, l_leg speeds
    33, 34,                   # r_leg speeds
    35, 36, 37, 38,           # asymmetries (4)
    47,                       # root_sway_xy_rms
]


def _diema_marker_names(diema_part: str) -> tuple[str, ...]:
    """Vicon marker *names* (not indices) for a DIEMA 6-part name."""
    out: list[str] = []
    for vg in DIEMA_TO_C3D_GROUPS[diema_part]:
        out.extend(VICON_PRODUCTION_PART_GROUPS[vg])
    return tuple(out)


def _diema_marker_names_for_parts(parts: tuple[str, ...]) -> tuple[str, ...]:
    if not parts:
        return ()
    names: list[str] = []
    for p in parts:
        names.extend(_diema_marker_names(p))
    return tuple(names)


def _marker_indices_in_clip(
    marker_names: tuple[str, ...], clip_labels: list[str],
) -> tuple[int, ...]:
    """Look up marker names → indices in *this clip's* label list.

    Tolerant: clips with 58 markers (some include an extra unnamed marker)
    are handled because we look up by name; we silently skip names that
    are missing from this clip (extremely rare in practice).
    """
    name_to_idx = {n: i for i, n in enumerate(clip_labels)}
    return tuple(sorted(name_to_idx[n] for n in marker_names if n in name_to_idx))


def _load_val_c3d(
    val_filenames: list[str], c3d_root: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[str]], list[float]]:
    """Read raw .c3d for the val performers; return per-clip artefacts.

    Returns ``(markers_list, residual_list, label_list, rate_list)``.
    ``markers_list[i]`` has shape ``(F_i, 57, 3)``.
    """
    markers_list: list[np.ndarray] = []
    residual_list: list[np.ndarray] = []
    label_list: list[list[str]] = []
    rate_list: list[float] = []
    for fn in val_filenames:
        # Strip BVH-style suffix if any; .c3d sits in same /raw/c3d/{train,test}/.
        cand = c3d_root / "train" / f"{fn}.c3d"
        if not cand.exists():
            cand = c3d_root / "test" / f"{fn}.c3d"
        if not cand.exists():
            raise FileNotFoundError(f"C3D not found for {fn!r} (tried {cand})")
        d = read_c3d(cand, strict=False)
        markers_list.append(d["markers"])
        residual_list.append(d["residual"])
        label_list.append(d["marker_labels"])
        rate_list.append(float(d["rate"]))
    return markers_list, residual_list, label_list, rate_list


def _apply_marker_mask(
    markers: np.ndarray,
    clip_labels: list[str],
    mask_marker_names: tuple[str, ...],
    mode: str,
    donor_markers_and_labels: list[tuple[np.ndarray, list[str]]] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a masked copy of ``markers`` of shape (F, M, 3).

    Masking is by marker *name*, not raw index, so a clip with 58 markers
    instead of 57 (a 2.5% minority in DIEM-A) is still masked correctly.

    For ``mode='zero'`` the masked-marker entries are set to ``np.nan`` so
    the downstream stats (NaN-safe) treat them as missing. Using NaN
    rather than 0 is more honest because a marker at the world origin
    would be a wild artefact, whereas NaN is what dropped-marker frames
    already look like — the natural perturbation that the stat code
    is designed to tolerate.

    For ``mode='shuffle'`` (post-C3 fix, 2026-06-12): we use a single
    donor clip per call and graft ALL the masked markers from that same
    donor onto the target. This matches the skeleton-side
    ``_apply_mask_inplace(mode='shuffle')`` semantics where one
    permutation index is shared across all masked joints, so the
    per-clip *part* (head, r_arm, …) is replaced as a coherent block
    rather than as N independent random markers — preserving within-
    part geometric coherence and matching Breiman/Fisher *grouped*
    permutation feature importance.

    The donor must have every marker name in ``mask_marker_names``; if
    the first candidate is missing any name we resample (up to 5 tries)
    and fall back to per-marker NaN on the missing names if the resample
    is exhausted.
    """
    out = markers.copy()
    if not mask_marker_names:
        return out
    mask_idx = _marker_indices_in_clip(mask_marker_names, clip_labels)
    if not mask_idx:
        return out
    if mode == "zero":
        for m in mask_idx:
            out[:, m, :] = np.nan
        return out
    if mode == "shuffle":
        if donor_markers_and_labels is None or rng is None:
            raise ValueError("shuffle mode requires donor_markers_and_labels and rng")
        F = out.shape[0]
        target_name_at_idx = clip_labels
        # Pick ONE donor clip that has all the masked markers; that single
        # donor will be the source for the whole masked part-group.
        chosen_donor: np.ndarray | None = None
        chosen_labels: list[str] | None = None
        for _ in range(5):
            k = int(rng.integers(0, len(donor_markers_and_labels)))
            donor, donor_labels = donor_markers_and_labels[k]
            target_marker_names = {target_name_at_idx[m] for m in mask_idx}
            if all(n in donor_labels for n in target_marker_names):
                chosen_donor = donor
                chosen_labels = donor_labels
                break

        if chosen_donor is None:
            # No donor had all the masked markers; per-marker best-effort
            # fallback (matches the legacy behaviour with NaN on missing).
            for m in mask_idx:
                marker_name = target_name_at_idx[m]
                # Try one more donor for this single marker.
                for _ in range(5):
                    k = int(rng.integers(0, len(donor_markers_and_labels)))
                    donor, donor_labels = donor_markers_and_labels[k]
                    if marker_name not in donor_labels:
                        continue
                    d_idx = donor_labels.index(marker_name)
                    F_donor = donor.shape[0]
                    if F_donor == F:
                        out[:, m, :] = donor[:, d_idx, :]
                    else:
                        idx = np.round(np.linspace(0, F_donor - 1, F)).astype(np.int64)
                        out[:, m, :] = donor[idx, d_idx, :]
                    break
                else:
                    out[:, m, :] = np.nan
            return out

        # Common path: one donor, used for every masked marker (grouped FI).
        F_donor = chosen_donor.shape[0]
        if F_donor == F:
            t_index = np.arange(F, dtype=np.int64)
        else:
            t_index = np.round(np.linspace(0, F_donor - 1, F)).astype(np.int64)
        for m in mask_idx:
            marker_name = target_name_at_idx[m]
            d_idx = chosen_labels.index(marker_name)
            out[:, m, :] = chosen_donor[t_index, d_idx, :]
        return out
    raise ValueError(f"unknown mode: {mode!r}")


def _recompute_stats_22d(
    markers: np.ndarray,
    marker_labels: list[str],
    rate: float,
    residual: np.ndarray | None,
) -> np.ndarray:
    """Recompute 22-D contact-only stats for one masked clip."""
    stats = compute_c3d_stats(
        markers, marker_labels, rate=rate, residual=residual,
    )
    full = stats_to_vector(stats)
    return full[CONTACT_FEATURES_22D].astype(np.float32)


def _run_fold(
    fold_id: int,
    num_folds: int,
    mask_mode: str,
    env: EnvConfig,
    c3d_root: Path,
    out_root: Path,
) -> dict | None:
    cache_npz = np.load(str(env.project_root / "output/c3d_stats_cache.npz"),
                        allow_pickle=True)
    all_filenames = np.asarray(cache_npz["filenames"])
    all_features_51d = np.asarray(cache_npz["features"], dtype=np.float32)

    # The cache covers BOTH train and test (~9.9k clips). Intersect with
    # the canonical train NPZ's filename set so the filter is provenance-
    # backed rather than a name heuristic (post-W3 fix 2026-06-12).
    train_npz = np.load(
        str(env.processed_dir / "motion_train_quat.npz"), allow_pickle=True,
    )
    canonical_train_names = set(str(fn) for fn in train_npz["filenames"])
    is_train = np.array([str(fn) in canonical_train_names for fn in all_filenames])
    cached_filenames = all_filenames[is_train]
    cached_features_51d = all_features_51d[is_train]

    # 22-D contact subset (un-masked, train-side; used to train the probe).
    cached_22d = cached_features_51d[:, CONTACT_FEATURES_22D]
    cached_labels = np.array(
        [emotion_to_label(str(fn)) for fn in cached_filenames], dtype=np.int64
    )

    # Train / val LPO split on the cached filename order.
    splits = generate_lpo_splits(cached_filenames.tolist(), num_folds=num_folds)
    train_indices = [idx for _, idx in splits[fold_id]["train"]]
    val_indices = [idx for _, idx in splits[fold_id]["val"]]
    val_filenames = [str(cached_filenames[i]) for i in val_indices]

    X_tr = cached_22d[train_indices]
    y_tr = cached_labels[train_indices]
    y_va = cached_labels[val_indices]
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(
        C=1.0, max_iter=2000, n_jobs=-1, solver="lbfgs", random_state=42,
    )
    clf.fit(scaler.transform(X_tr), y_tr)
    click.echo(f"  fold_{fold_id:02d} probe trained "
               f"(n_train={len(train_indices)}, n_val={len(val_indices)})")

    # Load val + a donor pool for shuffle (use the full val set as donor pool;
    # train clips would also work but val keeps the donor distribution closer
    # to the masked target distribution).
    t0 = time.time()
    val_markers, val_residual, val_marker_labels, val_rates = _load_val_c3d(
        val_filenames, c3d_root,
    )
    click.echo(f"  fold_{fold_id:02d} read {len(val_filenames)} .c3d in "
               f"{time.time()-t0:.1f}s")
    # Marker label sets differ in ~2.5% of clips (some have 58 markers
    # instead of the canonical 57). We mask by *name*, so the per-clip
    # label list is all we need.
    n_marker_variants = len({tuple(lbls) for lbls in val_marker_labels})
    if n_marker_variants > 1:
        click.echo(
            f"  fold_{fold_id:02d} note: {n_marker_variants} distinct marker "
            f"label sequences across val clips — masking by name preserves correctness"
        )

    # Mask-key set: baseline + 6 singles + cumulative top-k for 3 orderings.
    single_keys: list[tuple[str, ...]] = [()] + [(p,) for p in PARTS_ORDERED]

    fused_val_logits_by_key: dict[tuple[str, ...], np.ndarray] = {}

    def _logits_for_mask(key: tuple[str, ...]) -> np.ndarray:
        mask_marker_names = _diema_marker_names_for_parts(key)
        rng = np.random.default_rng(fold_id * 1000) if mask_mode == "shuffle" else None
        X_va = np.zeros((len(val_filenames), len(CONTACT_FEATURES_22D)),
                        dtype=np.float32)
        for i in range(len(val_filenames)):
            donors_and_labels = (
                [(val_markers[j], val_marker_labels[j])
                 for j in range(len(val_markers)) if j != i]
                if mask_mode == "shuffle" else None
            )
            masked = _apply_marker_mask(
                val_markers[i], val_marker_labels[i], mask_marker_names,
                mask_mode, donor_markers_and_labels=donors_and_labels, rng=rng,
            )
            X_va[i] = _recompute_stats_22d(
                masked, val_marker_labels[i], val_rates[i], val_residual[i],
            )
        proba = clf.predict_proba(scaler.transform(X_va))
        logits = np.full((len(X_va), NUM_CLASS), -50.0, dtype=np.float32)
        for j, cls in enumerate(clf.classes_):
            logits[:, int(cls)] = np.log(np.clip(proba[:, j], 1e-12, 1.0)).astype(np.float32)
        return logits

    for key in single_keys:
        t0 = time.time()
        fused_val_logits_by_key[key] = _logits_for_mask(key)
        f1 = _macro_f1(y_va, fused_val_logits_by_key[key].argmax(1))
        key_label = "baseline" if len(key) == 0 else "+".join(key)
        click.echo(f"    fold_{fold_id:02d} c3d mask {key_label:<20} "
                   f"F1={f1*100:.2f}%  ({time.time()-t0:.1f}s)")

    # Cumulative masks (only those not already computed via singles).
    base_f1 = _macro_f1(y_va, fused_val_logits_by_key[()].argmax(1))
    part_f1 = {p: _macro_f1(y_va, fused_val_logits_by_key[(p,)].argmax(1))
               for p in PARTS_ORDERED}
    drops = {p: base_f1 - part_f1[p] for p in PARTS_ORDERED}
    cumulative_keys, *_ = _resolve_mask_keys_for_curves(drops, seed=fold_id)
    needed = [k for k in cumulative_keys if k not in fused_val_logits_by_key]
    for key in needed:
        t0 = time.time()
        fused_val_logits_by_key[key] = _logits_for_mask(key)
        click.echo(f"    fold_{fold_id:02d} c3d mask {'+'.join(key):<20} "
                   f"({time.time()-t0:.1f}s)")

    # Save per-mask logits in the same naming scheme as externals.
    save_dir = out_root / "member_masked_logits" / "c3d" / f"fold_{fold_id:02d}_{mask_mode}"
    save_dir.mkdir(parents=True, exist_ok=True)
    for key, lg in fused_val_logits_by_key.items():
        key_name = "baseline" if len(key) == 0 else "_".join(sorted(key))
        np.save(save_dir / f"{key_name}.npy", lg.astype(np.float32))
    np.save(save_dir / "_labels.npy", y_va.astype(np.int64))
    np.save(save_dir / "_filenames.npy", np.asarray(val_filenames))

    record = {
        "member": "c3d",
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
@click.option("--fold", default="0", help="Fold number or 'all'")
@click.option("--num-folds", default=10, type=int)
@click.option("--mask-mode", default="shuffle", type=click.Choice(["zero", "shuffle"]))
@click.option("--c3d-root", default="data/diema_challenge/raw/c3d")
@click.option("--output-dir", default=None,
              help="Defaults to docs/analysis/ensemble_part_masking/")
def main(fold: str, num_folds: int, mask_mode: str, c3d_root: str,
         output_dir: str | None) -> None:
    env = EnvConfig()
    out_root = (Path(output_dir) if output_dir
                else env.project_root / "docs/analysis/ensemble_part_masking")
    out_root.mkdir(parents=True, exist_ok=True)
    c3d_path = Path(c3d_root)

    if fold == "all":
        fold_ids = list(range(num_folds))
    else:
        fold_ids = [int(fold)]

    records: list[dict] = []
    for f_id in fold_ids:
        click.echo(f"=== c3d fold_{f_id:02d} ({mask_mode}) ===")
        rec = _run_fold(f_id, num_folds, mask_mode, env, c3d_path, out_root)
        if rec is not None:
            records.append(rec)

    import json
    summary = {
        "member": "c3d",
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
    out_path = out_root / f"c3d_{mask_mode}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    click.echo(f"\nSaved {out_path}")
    click.echo(f"Baseline F1: {summary['baseline_f1_mean']*100:.2f} ± "
               f"{summary['baseline_f1_std']*100:.2f} %")
    for p in PARTS_ORDERED:
        click.echo(f"  {p:>6}: {summary['drops_by_part_mean'][p]*100:+.3f} pp")


if __name__ == "__main__":
    main()
