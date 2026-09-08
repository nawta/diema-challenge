""" fusion: combine PROD 7-way + skeleton-based externals → 10-way.

Related:
- tools/explain_ensemble_part_masking.py  (PROD per-mask logits cache)
- tools/explain_external_part_masking.py   (external per-mask logits cache)
- experiments/exp090_ensemble_patterns/build_logitmean_submission.py
    (canonical logit-mean fusion for the submitted ensemble)
- diema/data/parser::IDX_TO_EMOTION

The PROD tool caches per-(fold, mask_key, member) logits under
``docs/analysis/ensemble_part_masking/member_masked_logits/prod/fold_NN_<mode>/``
as ``{key_name}__{exp_name}.npy``. The external tool caches under
``docs/analysis/ensemble_part_masking/member_masked_logits/<member>/fold_NN_<mode>/``
as ``{key_name}.npy`` (one file per mask key, no per-member suffix since
each external is its own member).

External logits are stored as ``log p`` from the sklearn ``predict_proba``.
Since ``softmax(log p) == p`` for a valid probability distribution,
treating those log-probs as "raw logits" and re-feeding through
``softmax(log(softmax(L))).mean(0)`` recovers ``log p``. The fusion math
is invariant.

Usage::

    python tools/fuse_ensemble_part_masking.py --mode shuffle \\
        --include motionbert,mamp60,mamp120 \\
        --num-folds 10

Outputs:
- ``docs/analysis/ensemble_part_masking/ensemble_10way_<mode>_fold{NN}_faithfulness.json``
- ``docs/analysis/ensemble_part_masking/ensemble_10way_<mode>_part_importance.csv``
- ``docs/analysis/ensemble_part_masking/ensemble_10way_<mode>_aucs.csv``
- ``docs/analysis/ensemble_part_masking/ensemble_10way_<mode>_part_heatmap.png``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diema.data.parser import IDX_TO_EMOTION  # noqa: E402

from tools.explain_ensemble_part_masking import (  # noqa: E402
    PARTS_ORDERED,
    PROD_MEMBERS,
    _ensemble_macro_f1,
    _logit_mean_fuse_argmax,
    _orderings_for_drops,
    _resolve_mask_keys_for_curves,
    _faithfulness_curve_from_cache,
    _mask_key,
)
from tools.explain_part_masking import _auc_drop, _macro_f1, _per_class_f1  # noqa: E402


NUM_CLASS = 12


def _load_prod_logits_for_fold(
    cache_root: Path, fold_id: int, mask_mode: str,
) -> tuple[dict[tuple[str, ...], dict[str, np.ndarray]], np.ndarray]:
    """Load per-(mask_key, prod_member) logits for one fold.

    Returns ``(per_mask_member_logits, labels)``. ``per_mask_member_logits[key][member]``
    has shape ``(N_val, NUM_CLASS)``.
    """
    fold_dir = cache_root / "prod" / f"fold_{fold_id:02d}_{mask_mode}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"PROD cache missing: {fold_dir}")
    labels = np.load(fold_dir / "_labels.npy")
    by_key: dict[tuple[str, ...], dict[str, np.ndarray]] = {}
    for npy in sorted(fold_dir.glob("*.npy")):
        if npy.name.startswith("_"):
            continue
        stem = npy.stem  # e.g. "baseline__exp002_a04_smooth" or "head__exp003_..."
        key_name, _sep, exp_name = stem.partition("__")
        if not exp_name:
            raise RuntimeError(f"unexpected cache filename: {npy.name}")
        if key_name == "baseline":
            key: tuple[str, ...] = ()
        else:
            key = tuple(sorted(key_name.split("_")
                               if "_" not in key_name
                               else _split_compound_key(key_name)))
        # Reconstruct the tuple key from the canonical name format.
        key = _stem_to_mask_key(key_name)
        by_key.setdefault(key, {})[exp_name] = np.load(npy)
    return by_key, labels


def _split_compound_key(key_name: str) -> list[str]:
    """Recover the part names from a stem of "head_l_arm_r_leg" style."""
    # Each underscore-joined part name itself uses underscores (l_arm, r_leg
    # etc.). We rely on the canonical part-name set to greedily match.
    parts: list[str] = []
    rest = key_name
    while rest:
        matched = False
        for p in PARTS_ORDERED:
            if rest == p:
                parts.append(p); rest = ""; matched = True; break
            if rest.startswith(p + "_"):
                parts.append(p); rest = rest[len(p) + 1:]; matched = True; break
        if not matched:
            raise RuntimeError(f"can't parse mask key stem: {key_name!r}")
    return parts


def _stem_to_mask_key(key_name: str) -> tuple[str, ...]:
    """"baseline" → (), "head" → ("head",), "head_l_arm" → ("head", "l_arm")."""
    if key_name == "baseline":
        return ()
    return tuple(sorted(_split_compound_key(key_name)))


def _load_external_logits_for_fold(
    cache_root: Path, member: str, fold_id: int, mask_mode: str,
) -> dict[tuple[str, ...], np.ndarray]:
    """Load per-mask logits for one external member, one fold."""
    fold_dir = cache_root / member / f"fold_{fold_id:02d}_{mask_mode}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"{member} cache missing: {fold_dir}")
    by_key: dict[tuple[str, ...], np.ndarray] = {}
    for npy in sorted(fold_dir.glob("*.npy")):
        if npy.name.startswith("_"):
            continue
        stem = npy.stem
        by_key[_stem_to_mask_key(stem)] = np.load(npy)
    return by_key


def _run_fold(
    fold_id: int,
    cache_root: Path,
    mask_mode: str,
    include_externals: list[str],
    out_root: Path,
) -> dict | None:
    """Compute Nway ensemble (7 PROD + included externals) F1 for this fold."""
    prod_by_key, prod_labels = _load_prod_logits_for_fold(
        cache_root, fold_id, mask_mode,
    )
    ext_by_key: dict[str, dict[tuple[str, ...], np.ndarray]] = {}
    for m in include_externals:
        ext_by_key[m] = _load_external_logits_for_fold(
            cache_root, m, fold_id, mask_mode,
        )

    # Verify alignment between PROD and external val sets.
    ext_first = include_externals[0] if include_externals else None
    if ext_first is not None:
        ext_labels = np.load(
            cache_root / ext_first / f"fold_{fold_id:02d}_{mask_mode}" / "_labels.npy"
        )
        if not np.array_equal(prod_labels, ext_labels):
            raise RuntimeError(
                f"fold {fold_id}: PROD labels disagree with {ext_first} labels"
            )

    # Build the set of mask keys we have for all members in this fold.
    # Member-side cumulative orderings differ (each member's "important_first"
    # is computed from its OWN drops), so intersecting across members usually
    # leaves only the baseline + 6 singles + the all-parts mask in common.
    # That is sufficient for the headline ensemble drops; cumulative AUC at
    # k>1 is reported when available.
    keys = set(prod_by_key.keys())
    for m in include_externals:
        keys = keys.intersection(set(ext_by_key[m].keys()))

    N_ENS = len(PROD_MEMBERS) + len(include_externals)
    fused_f1_by_key: dict[tuple[str, ...], float] = {}
    per_class_by_key: dict[tuple[str, ...], np.ndarray] = {}
    for key in keys:
        member_logits = [prod_by_key[key][m] for m in PROD_MEMBERS]
        for m in include_externals:
            member_logits.append(ext_by_key[m][key])
        assert len(member_logits) == N_ENS
        pred = _logit_mean_fuse_argmax(member_logits)
        fused_f1_by_key[key] = _macro_f1(prod_labels, pred)
        per_class_by_key[key] = _per_class_f1(prod_labels, pred)

    if () not in fused_f1_by_key or any((p,) not in fused_f1_by_key for p in PARTS_ORDERED):
        click.echo(
            f"  [skip-fold] fold_{fold_id:02d}: missing baseline or some single "
            f"part keys after intersection (have {sorted(fused_f1_by_key.keys())[:5]}…)",
            err=True,
        )
        return None

    base_f1 = fused_f1_by_key[()]
    base_per_class = per_class_by_key[()]

    part_f1 = {p: fused_f1_by_key[(p,)] for p in PARTS_ORDERED}
    drops_by_part = {p: base_f1 - part_f1[p] for p in PARTS_ORDERED}
    important, reverse, random_order = _orderings_for_drops(drops_by_part, seed=fold_id)

    # Best-effort faithfulness curves: walk the cumulative ordering and stop
    # at the first k where the required mask key isn't in the intersection.
    def _safe_curve(order: list[str]) -> tuple[list[float], int]:
        curve = [base_f1]
        for k in range(1, len(order) + 1):
            key = _mask_key(tuple(order[:k]))
            if key not in fused_f1_by_key:
                return curve, k - 1
            curve.append(fused_f1_by_key[key])
        return curve, len(order)

    curves: dict[str, list[float]] = {}
    curve_depths: dict[str, int] = {}
    for name, order in (("important_first", important),
                        ("reverse", reverse), ("random", random_order)):
        c, depth = _safe_curve(order)
        curves[name] = c
        curve_depths[name] = depth
    aucs = {name: _auc_drop(c) for name, c in curves.items()}
    min_depth = min(curve_depths.values())

    # Simple "single-mask" proxy for the gap when cumulative is incomplete.
    drop_max = max(drops_by_part.values())
    drop_min = min(drops_by_part.values())
    single_gap = drop_max - drop_min

    click.echo(
        f"fold_{fold_id:02d}  N_ENS={N_ENS}  baseline {base_f1*100:.2f}%  "
        f"drops max={drop_max*100:.2f}%, min={drop_min*100:.2f}%, "
        f"single-mask gap={single_gap*100:+.2f}pp  "
        f"(curve depth k≤{min_depth})"
    )

    record = {
        "ensemble": f"{N_ENS}way_logitmean",
        "mask_mode": mask_mode,
        "fold": fold_id,
        "n_members": N_ENS,
        "members_prod": list(PROD_MEMBERS),
        "members_external": list(include_externals),
        "baseline_f1": base_f1,
        "part_f1": {p: float(part_f1[p]) for p in PARTS_ORDERED},
        "drops_by_part": {p: float(drops_by_part[p]) for p in PARTS_ORDERED},
        "order_important": important,
        "order_reverse": reverse,
        "order_random": random_order,
        "curves": curves,
        "curve_depths": curve_depths,
        "aucs_partial": aucs,
        "single_mask_gap": single_gap,
        "drop_max_part": max(drops_by_part, key=drops_by_part.get),
        "drop_min_part": min(drops_by_part, key=drops_by_part.get),
        "n_val": int(prod_labels.shape[0]),
    }
    name = f"ensemble_{N_ENS}way_{mask_mode}_fold{fold_id:02d}_faithfulness.json"
    (out_root / name).write_text(json.dumps(record, indent=2))

    rows = []
    for p in PARTS_ORDERED:
        row = {
            "ensemble": f"{N_ENS}way_logitmean",
            "mask_mode": mask_mode,
            "fold": fold_id,
            "part": p,
            "baseline_f1": base_f1,
            "masked_f1": part_f1[p],
            "f1_drop": drops_by_part[p],
        }
        for c in range(NUM_CLASS):
            row[f"f1_drop_{IDX_TO_EMOTION[c]}"] = float(
                base_per_class[c] - per_class_by_key[(p,)][c]
            )
        rows.append(row)
    record["_csv_rows"] = rows
    return record


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@click.command()
@click.option("--mode", default="shuffle", type=click.Choice(["zero", "shuffle"]))
@click.option("--include", default="motionbert,mamp60,mamp120,c3d",
              help=("Comma-separated external members to include "
                    "(default = full 11-way: all 3 skeleton externals + C3D)"))
@click.option("--num-folds", default=10, type=int)
@click.option("--cache-root", default=None,
              help="Override member_masked_logits cache root")
@click.option("--out-root", default=None,
              help="Override output dir for fusion summary")
def main(mode: str, include: str, num_folds: int,
         cache_root: str | None, out_root: str | None) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.env import EnvConfig
    env = EnvConfig()
    base = env.project_root / "docs/analysis/ensemble_part_masking"
    cache = Path(cache_root) if cache_root else base / "member_masked_logits"
    out = Path(out_root) if out_root else base
    out.mkdir(parents=True, exist_ok=True)

    include_list = [m.strip() for m in include.split(",") if m.strip()]
    click.echo(f"Fusion: PROD ({len(PROD_MEMBERS)}) + externals ({include_list}) "
               f"= {len(PROD_MEMBERS) + len(include_list)}-way, mode={mode}")

    aggregate_aucs: list[dict] = []
    all_rows: list[dict] = []
    for fid in range(num_folds):
        rec = _run_fold(fid, cache, mode, include_list, out)
        if rec is None:
            continue
        all_rows.extend(rec.pop("_csv_rows"))
        aggregate_aucs.append({
            "fold": fid,
            "baseline_f1": rec["baseline_f1"],
            "single_mask_gap": rec["single_mask_gap"],
            "drop_max_part": rec["drop_max_part"],
            "drop_min_part": rec["drop_min_part"],
            **{f"auc_{k}": v for k, v in rec["aucs_partial"].items()},
            "auc_gap_partial": rec["aucs_partial"]["important_first"]
            - rec["aucs_partial"]["reverse"],
        })

    if not all_rows:
        click.echo("No folds produced; aborting.")
        return

    N_ENS = len(PROD_MEMBERS) + len(include_list)
    df = pd.DataFrame(all_rows)
    csv_path = out / f"ensemble_{N_ENS}way_{mode}_part_importance.csv"
    df.to_csv(csv_path, index=False)
    auc_df = pd.DataFrame(aggregate_aucs)
    auc_path = out / f"ensemble_{N_ENS}way_{mode}_aucs.csv"
    auc_df.to_csv(auc_path, index=False)

    single_gap = np.array([r["single_mask_gap"] for r in aggregate_aucs])
    partial_gap = np.array([r["auc_gap_partial"] for r in aggregate_aucs])
    base = np.array([r["baseline_f1"] for r in aggregate_aucs])
    click.echo(
        f"\n[ensemble {N_ENS}-way {mode}]"
        f"\n  baseline F1 per-fold:   {base.mean()*100:.2f} ± {base.std(ddof=1)*100:.2f} %"
        f"\n  single-mask gap (top1 − bottom1 drop):  "
        f"{single_gap.mean()*100:+.2f} ± {single_gap.std(ddof=1)*100:.2f} pp "
        f"(n={len(single_gap)}, sign {int((single_gap>0).sum())}/{len(single_gap)})"
        f"\n  partial-AUC gap (best-effort cumulative): "
        f"{partial_gap.mean()*100:+.2f} ± {partial_gap.std(ddof=1)*100:.2f} pp"
    )
    # Top-1 part voting across folds.
    from collections import Counter
    top1_counts = Counter(r["drop_max_part"] for r in aggregate_aucs)
    click.echo(f"  Top-1 most important part across folds: {dict(top1_counts)}")
    click.echo(f"\nSaved CSVs: {csv_path}, {auc_path}")

    # Heatmap
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
            f"Ensemble {N_ENS}-way logit-mean ({mode}): F1 drop per body part"
        )
        fig.colorbar(im, ax=ax, label="F1 drop (baseline − masked)")
        fig.tight_layout()
        fig_path = out / f"ensemble_{N_ENS}way_{mode}_part_heatmap.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        click.echo(f"Heatmap: {fig_path}")
    except Exception as exc:
        click.echo(f"[warn] heatmap skipped: {exc}", err=True)


if __name__ == "__main__":
    main()
