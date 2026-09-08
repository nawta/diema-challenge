""" exp068 paired bootstrap for the best multi-crop setting.

Re-runs multi-crop inference for ONE specific (policy, num_crops, aggregation)
combination across all 10 folds × 7 ensemble members, builds per-sample
aggregated OOF logits, then computes a sample-level paired bootstrap
(Macro-F1 of multi-crop minus Macro-F1 of single-crop baseline) using the
existing saved single-crop OOF logits.

Why a separate script: ``tools/infer_test_multicrop.py`` does not save
aggregated OOF logits to disk (only summary metrics), so re-running here is
cheaper than modifying that tool and rerunning the full 24-setting sweep.

CLI usage::

    python tools/multicrop_bootstrap.py \
        --policy phase_shift \
        --num-crops 5 \
        --aggregation logit_mean \
        --bootstrap-iter 1000

See: §5.1 / §11.2
Related: tools/infer_test_multicrop.py (full sweep),
         tools/calibrate_ensemble.py (paired_bootstrap_f1_diff,
         load_ensemble_oof helpers reused).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pybvh_ml  # noqa: E402

from utils.env import EnvConfig  # noqa: E402
from diema.data.parser import IDX_TO_EMOTION  # noqa: E402
from diema.data.splits import generate_lpo_splits  # noqa: E402
from diema.models import build_model  # noqa: E402
import diema.models  # noqa: F401, E402

from tools.calibrate_ensemble import (  # noqa: E402
    DEFAULT_ENSEMBLE,
    load_ensemble_oof,
    macro_f1,
    paired_bootstrap_f1_diff,
    per_class_f1,
)
from tools.infer_test_ensemble import (  # noqa: E402
    _cfg_to_builder_ns,
    _load_checkpoint_into_model,
)
from tools.infer_test_multicrop import (  # noqa: E402
    aggregate_member_fold,
    run_inference_for_fold,
)


@click.command()
@click.option("--policy", required=True,
              type=click.Choice(
                  ["phase_shift", "uniform_endpoints",
                   "center_quartiles", "motion_energy_softmax"],
              ))
@click.option("--num-crops", required=True, type=int)
@click.option("--aggregation", required=True,
              type=click.Choice(["logit_mean", "prob_mean"]))
@click.option("--motion-energy-tau", default=2.0, type=float)
@click.option("--bootstrap-iter", default=1000, type=int)
@click.option("--bootstrap-seed", default=42, type=int)
@click.option("--device", default="cuda",
              type=click.Choice(["cuda", "cpu"]))
@click.option(
    "--out-dir",
    default="experiments/exp068_multicrop_inference/analysis",
    type=click.Path(file_okay=False),
)
def main(
    policy: str,
    num_crops: int,
    aggregation: str,
    motion_energy_tau: float,
    bootstrap_iter: int,
    bootstrap_seed: int,
    device: str,
    out_dir: str,
) -> None:
    env = EnvConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device if (torch.cuda.is_available() or device == "cpu") else "cpu")
    click.echo(f"Inference device: {dev}")
    click.echo(f"Setting: policy={policy}, K={num_crops}, agg={aggregation}")

    # --- Single-crop baseline (saved OOF) -----------------------------------
    folds_baseline = load_ensemble_oof(
        env.artifacts_dir, DEFAULT_ENSEMBLE, num_folds=10,
    )
    base_logits_per_fold = [f.logits for f in folds_baseline]
    base_labels_per_fold = [f.labels for f in folds_baseline]
    base_filenames_per_fold = [f.filenames for f in folds_baseline]
    base_logits_all = np.concatenate(base_logits_per_fold, axis=0)
    base_labels_all = np.concatenate(base_labels_per_fold, axis=0)
    base_preds_all = base_logits_all.argmax(axis=1)
    base_f1 = macro_f1(base_preds_all, base_labels_all)
    click.echo(f"Single-crop 7-way baseline F1: {base_f1*100:.3f}%")

    # --- Multi-crop inference for the requested setting ---------------------
    data_path = env.processed_dir / "motion_train_quat.npz"
    preprocessed = pybvh_ml.load_preprocessed(str(data_path))
    all_filenames_dataset = preprocessed["filenames"]
    all_clips = preprocessed["clips"]
    all_labels = preprocessed["labels"]
    skeleton_info = preprocessed.get("skeleton_info", {}) or {}
    dataset_euler_orders = (
        skeleton_info.get("euler_orders") if isinstance(skeleton_info, dict) else None
    )
    splits = generate_lpo_splits(all_filenames_dataset, num_folds=10)

    # Per-(member, fold) aggregated logits.
    per_member_per_fold: dict[str, dict[int, dict]] = {}
    for exp in DEFAULT_ENSEMBLE:
        per_member_per_fold[exp] = {}
        exp_dir = env.artifacts_dir / exp
        for fold in range(10):
            ckpt_path = exp_dir / f"fold_{fold:02d}" / "best.ckpt"
            cfg_path = exp_dir / f"fold_{fold:02d}" / "config.yaml"
            if not ckpt_path.exists():
                click.echo(f"[skip] {exp}/fold_{fold:02d}: ckpt missing", err=True)
                continue
            cfg_dict = OmegaConf.to_container(OmegaConf.load(str(cfg_path)), resolve=True)
            target_repr = cfg_dict.get("target_repr", "6d")
            clip_length = cfg_dict.get("clip_length", 64)

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
                f"  {exp} fold_{fold:02d}: {len(val_indices)} samples"
            )
            run_out = run_inference_for_fold(
                model=model,
                val_indices=val_indices,
                val_filenames=val_filenames,
                val_labels=val_labels,
                clips=all_clips,
                target_repr=target_repr,
                clip_length=clip_length,
                policies=(policy,),
                num_crops_list=(num_crops,),
                device=dev,
                batch_size=64,
                motion_energy_tau=motion_energy_tau,
                euler_orders=dataset_euler_orders,
            )
            conf = run_out["configs"][(policy, num_crops)]
            agg_logits = aggregate_member_fold(
                conf["logits"], conf["weights"], aggregation,
            )
            per_member_per_fold[exp][fold] = {
                "logits": agg_logits,
                "labels": val_labels,
                "filenames": val_filenames,
            }
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Equal-weight ensemble across members per fold ----------------------
    per_fold_logits = []
    per_fold_labels = []
    per_fold_filenames = []
    for fold in range(10):
        members_in_fold = [
            per_member_per_fold[exp][fold]
            for exp in DEFAULT_ENSEMBLE
            if fold in per_member_per_fold.get(exp, {})
        ]
        if len(members_in_fold) != len(DEFAULT_ENSEMBLE):
            click.echo(
                f"[warn] fold {fold:02d}: only {len(members_in_fold)}/"
                f"{len(DEFAULT_ENSEMBLE)} members present",
                err=True,
            )
            continue
        stacked = np.stack(
            [m["logits"] for m in members_in_fold], axis=0,
        )  # (M, N, num_classes)
        ensemble_logits = stacked.mean(axis=0)
        per_fold_logits.append(ensemble_logits)
        per_fold_labels.append(members_in_fold[0]["labels"])
        per_fold_filenames.append(members_in_fold[0]["filenames"])

    mc_logits_all = np.concatenate(per_fold_logits, axis=0)
    mc_labels_all = np.concatenate(per_fold_labels, axis=0)
    mc_preds_all = mc_logits_all.argmax(axis=1)
    mc_f1 = macro_f1(mc_preds_all, mc_labels_all)
    click.echo(f"Multi-crop F1: {mc_f1*100:.3f}% (Δ vs baseline {(mc_f1 - base_f1)*100:+.3f}pp)")

    # --- Verify sample alignment between baseline and multi-crop OOF --------
    mc_filenames_all = [
        fn for chunk in per_fold_filenames for fn in chunk
    ]
    base_filenames_all = [
        fn for chunk in base_filenames_per_fold for fn in chunk
    ]
    if mc_filenames_all != base_filenames_all:
        # Same per-fold filename order from generate_lpo_splits → same total order.
        raise RuntimeError(
            "Sample alignment differs between baseline and multi-crop OOF. "
            "Each fold should iterate val_pairs in identical order."
        )
    if not np.array_equal(mc_labels_all, base_labels_all):
        raise RuntimeError("label alignment differs between baseline and multi-crop OOF")

    # --- Sample-level paired bootstrap --------------------------------------
    boot = paired_bootstrap_f1_diff(
        base_preds_all, mc_preds_all, base_labels_all,
        n_iter=bootstrap_iter, seed=bootstrap_seed,
    )
    click.echo(
        f"Paired bootstrap diff (multi-crop − baseline): "
        f"mean = {boot['mean_diff_pp']:+.3f}pp, "
        f"95% CI [{boot['ci_2.5_pp']:+.3f}, {boot['ci_97.5_pp']:+.3f}]"
    )

    # --- Per-class F1 deltas ------------------------------------------------
    base_pcf1 = per_class_f1(base_preds_all, base_labels_all)
    mc_pcf1 = per_class_f1(mc_preds_all, mc_labels_all)
    click.echo("\nPer-class F1 (baseline → multi-crop):")
    for c in range(12):
        click.echo(
            f"  {IDX_TO_EMOTION[c]:<12}: {base_pcf1[c]*100:5.2f} → {mc_pcf1[c]*100:5.2f} "
            f"({(mc_pcf1[c]-base_pcf1[c])*100:+.2f}pp)"
        )

    # --- Per-fold std (criterion 2) -----------------------------------------
    base_per_fold_f1 = [
        macro_f1(L.argmax(axis=1), Y)
        for L, Y in zip(base_logits_per_fold, base_labels_per_fold)
    ]
    mc_per_fold_f1 = [
        macro_f1(L.argmax(axis=1), Y)
        for L, Y in zip(per_fold_logits, per_fold_labels)
    ]
    base_std = float(np.std(base_per_fold_f1, ddof=1))
    mc_std = float(np.std(mc_per_fold_f1, ddof=1))
    std_ratio = mc_std / base_std if base_std > 0 else float("inf")
    click.echo(
        f"\nPer-fold F1 std: baseline {base_std*100:.3f}pp → "
        f"multi-crop {mc_std*100:.3f}pp (ratio {std_ratio:.3f})"
    )

    # --- Verdict ------------------------------------------------------------
    delta_pp = (mc_f1 - base_f1) * 100
    c1 = delta_pp >= 0.30
    c2 = std_ratio <= 1.50
    c3 = boot["ci_2.5_pp"] > -0.10
    verdict = "GO" if (c1 and c2 and c3) else "NO-GO"
    click.echo(
        f"\nGo criteria: c1 (Δ≥+0.30pp) {'✓' if c1 else '✗'}, "
        f"c2 (std ratio ≤ 1.5) {'✓' if c2 else '✗'}, "
        f"c3 (CI lower > -0.10pp) {'✓' if c3 else '✗'} → {verdict}"
    )

    # Persist
    setting_tag = f"{policy}_K{num_crops}_{aggregation}"
    out_path = out / f"bootstrap_{setting_tag}.json"
    with open(out_path, "w") as fh:
        json.dump({
            "policy": policy,
            "num_crops": int(num_crops),
            "aggregation": aggregation,
            "baseline_f1": float(base_f1),
            "multicrop_f1": float(mc_f1),
            "delta_f1_pp": float(delta_pp),
            "bootstrap": boot,
            "per_fold_f1_baseline": list(map(float, base_per_fold_f1)),
            "per_fold_f1_multicrop": list(map(float, mc_per_fold_f1)),
            "per_fold_std_baseline_pp": float(base_std * 100),
            "per_fold_std_multicrop_pp": float(mc_std * 100),
            "per_fold_std_ratio": float(std_ratio),
            "per_class_f1_baseline": {IDX_TO_EMOTION[c]: float(base_pcf1[c]) for c in range(12)},
            "per_class_f1_multicrop": {IDX_TO_EMOTION[c]: float(mc_pcf1[c]) for c in range(12)},
            "per_class_f1_delta_pp": {
                IDX_TO_EMOTION[c]: float((mc_pcf1[c] - base_pcf1[c]) * 100) for c in range(12)
            },
            "criterion_1_plus_0.30pp": bool(c1),
            "criterion_2_std_ratio_within_1.5": bool(c2),
            "criterion_3_ci_lower_above_minus_0.10pp": bool(c3),
            "verdict": verdict,
        }, fh, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))
    click.echo(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
