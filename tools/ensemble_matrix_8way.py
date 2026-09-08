""" exp069 8-way ablation matrix.

Builds 5 ensemble configurations from existing OOF logits + the new
exp069 (Conv1D+Tr a04 reg @ max_epochs=40), and compares each to the
production 7-way single-crop baseline via sample-level paired bootstrap.

Configurations (§11.3):

  | tag | members | rationale |
  |-----|---------|-----------|
  | 7-way (baseline) | a01 + 6 other production members | current production |
  | 8-way-naive  | 7-way + exp069 (a04@40) | additive diversity |
  | 8-way-swap   | 7-way - a01 + exp069 (a04@40) | a04 replaces a01 |
  | 8-way-diverse | 7-way + exp069 (a04@40) | identical members to naive but
                   we treat the result as the "kept-as-distinct hyperparam"
                   variant — included for completeness with §11.3 |
  | 9-way | 7-way + exp069 + exp020 a00 | adds a 3rd Conv1D+Tr regularisation |

Note: 8-way-diverse and 8-way-naive use the same member set. We compute
both for symmetry with the plan's matrix, but expect identical numbers.

CLI usage::

    python tools/ensemble_matrix_8way.py --bootstrap-iter 1000

Outputs land under ``experiments/exp069_convtr_a04_40epoch/analysis/``:
  * matrix_results.json — per-config F1, Δ vs 7-way, paired bootstrap CI
  * verdict_inputs.json — Go-criterion checks for each config

See: §5.2 / §11.3
Related: tools/ensemble_models.py (single-config OOF averaging),
         tools/calibrate_ensemble.py (paired bootstrap helper).
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.env import EnvConfig  # noqa: E402
from diema.data.parser import IDX_TO_EMOTION  # noqa: E402

from tools.calibrate_ensemble import (  # noqa: E402
    DEFAULT_ENSEMBLE,
    load_member_oof,
    macro_f1,
    paired_bootstrap_f1_diff,
    per_class_f1,
)


# Production 7-way ensemble (matches calibrate_ensemble.DEFAULT_ENSEMBLE).
PRODUCTION_7WAY = list(DEFAULT_ENSEMBLE)

# New 8th member (exp069 a04@40).
EXP069_A04_40 = "exp069_convtr_a04_40epoch_a00"

# 9th member (exp020 a00 = drop=0.2 / ls=0.1 / max_epochs=80; the *third*
# Conv1D+Tr regularisation strength in addition to a01 and a04).
EXP020_A00 = "exp020_conv1d_transformer_a00"

# Member identifier for production a01 (used for the 'swap' config).
EXP020_A01 = "exp020_conv1d_transformer_a01"


def _ensemble_oof(
    artifacts_dir: Path, members: list[str], num_folds: int = 10,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Equal-weight average OOF logits across members; return concatenated
    (logits, labels, filenames) over all folds in ascending fold order.

    Asserts label/filename order matches across members per fold.
    """
    if not members:
        raise ValueError("members must be non-empty")
    per_member = [
        load_member_oof(artifacts_dir, m, num_folds=num_folds) for m in members
    ]
    out_logits = []
    out_labels = []
    out_filenames: list[str] = []
    for f in range(num_folds):
        ref_logits, ref_labels, ref_fnames = per_member[0][f]
        stack = [ref_logits]
        for mi, mem in enumerate(per_member[1:], start=1):
            li, la, fn = mem[f]
            if fn != ref_fnames:
                raise ValueError(
                    f"fold {f:02d}: filename order differs between "
                    f"{members[0]} and {members[mi]}"
                )
            if not np.array_equal(la, ref_labels):
                raise ValueError(
                    f"fold {f:02d}: labels differ between "
                    f"{members[0]} and {members[mi]}"
                )
            stack.append(li)
        ens = np.stack(stack, axis=0).mean(axis=0).astype(np.float64)
        out_logits.append(ens)
        out_labels.append(ref_labels.astype(np.int64))
        out_filenames.extend(ref_fnames)
    return np.concatenate(out_logits, axis=0), np.concatenate(out_labels, axis=0), out_filenames


def _per_fold_f1(
    artifacts_dir: Path, members: list[str], num_folds: int = 10,
) -> list[float]:
    """Per-fold Macro-F1 of the equal-weight ensemble over ``members``."""
    per_member = [
        load_member_oof(artifacts_dir, m, num_folds=num_folds) for m in members
    ]
    out: list[float] = []
    for f in range(num_folds):
        ref_logits, ref_labels, _ = per_member[0][f]
        stack = [ref_logits]
        for mi, mem in enumerate(per_member[1:], start=1):
            li, _, _ = mem[f]
            stack.append(li)
        mean_logits = np.stack(stack, axis=0).mean(axis=0)
        out.append(macro_f1(mean_logits.argmax(axis=1), ref_labels))
    return out


@click.command()
@click.option("--num-folds", default=10, type=int)
@click.option("--bootstrap-iter", default=1000, type=int)
@click.option("--bootstrap-seed", default=42, type=int)
@click.option(
    "--out-dir",
    default="experiments/exp069_convtr_a04_40epoch/analysis",
    type=click.Path(file_okay=False),
)
def main(num_folds: int, bootstrap_iter: int, bootstrap_seed: int, out_dir: str) -> None:
    env = EnvConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if num_folds < 1:
        raise click.BadParameter("--num-folds must be >= 1")
    if bootstrap_iter < 1:
        raise click.BadParameter("--bootstrap-iter must be >= 1")

    # --- Verify all member directories exist ----------------------------------
    members_to_check = (
        PRODUCTION_7WAY + [EXP069_A04_40, EXP020_A00]
    )
    missing = []
    for m in members_to_check:
        for f in range(num_folds):
            p = env.artifacts_dir / m / f"fold_{f:02d}" / "oof_logits.npy"
            if not p.exists():
                missing.append(str(p))
    if missing:
        raise click.ClickException(
            "Missing OOF logits:\n  " + "\n  ".join(missing[:10])
            + (f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else "")
        )

    # --- Compute SHA256 of every input OOF logits file ---
    # This lets the JSON output detect "stale matrix vs fresh inputs" — a
    # regression that bit us once when a corrupted fold was redone *after*
    # the matrix was computed (review2 C1 in the exp069 commit history).
    input_hashes: dict[str, str] = {}
    for m in members_to_check:
        for f in range(num_folds):
            p = env.artifacts_dir / m / f"fold_{f:02d}" / "oof_logits.npy"
            h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            input_hashes[f"{m}/fold_{f:02d}"] = h

    # --- Define configurations ------------------------------------------------
    seven_way = list(PRODUCTION_7WAY)
    if EXP020_A01 not in seven_way:
        raise click.ClickException(
            f"Cannot construct 8-way-swap: {EXP020_A01} is not in the 7-way baseline. "
            "PRODUCTION_7WAY may have been edited; update the swap config."
        )
    eight_way_naive = seven_way + [EXP069_A04_40]
    eight_way_swap = [m for m in seven_way if m != EXP020_A01] + [EXP069_A04_40]
    nine_way = seven_way + [EXP069_A04_40, EXP020_A00]

    # NOTE: plan §11.3 lists "8-way-diverse" as a separate row, but its member
    # set is identical to "8-way-naive" (both keep a01 + add a04@40). We
    # therefore compute one row and re-label its results in the JSON output
    # for symmetry with the plan's table. This avoids duplicate compute and
    # duplicate bootstrap iterations.
    configs = [
        ("7-way (baseline)", seven_way),
        ("8-way-naive", eight_way_naive),
        ("8-way-swap", eight_way_swap),
        ("9-way", nine_way),
    ]

    click.echo("=== Computing ensemble configs ===")
    config_results: dict[str, dict] = {}
    baseline_logits = None
    baseline_labels = None
    baseline_filenames: list[str] = []
    baseline_per_fold_f1: list[float] = []
    for tag, members in configs:
        logits, labels, fnames = _ensemble_oof(env.artifacts_dir, members, num_folds=num_folds)
        preds = logits.argmax(axis=1)
        overall_f1 = macro_f1(preds, labels)
        per_fold = _per_fold_f1(env.artifacts_dir, members, num_folds=num_folds)
        per_class = per_class_f1(preds, labels)
        click.echo(
            f"  {tag:<22}: F1 = {overall_f1*100:.3f}% "
            f"(per-fold mean {np.mean(per_fold)*100:.3f} ± {np.std(per_fold, ddof=1)*100:.3f}, "
            f"members={len(members)})"
        )
        config_results[tag] = {
            "members": members,
            "n_members": len(members),
            "logits": logits,
            "labels": labels,
            "filenames": fnames,
            "preds": preds,
            "overall_f1": overall_f1,
            "overall_acc": float((preds == labels).mean()),
            "per_fold_f1": per_fold,
            "per_fold_f1_mean": float(np.mean(per_fold)),
            "per_fold_f1_std": float(np.std(per_fold, ddof=1)),
            "per_class_f1": {IDX_TO_EMOTION[c]: float(per_class[c]) for c in range(12)},
        }
        if tag == "7-way (baseline)":
            baseline_logits = logits
            baseline_labels = labels
            baseline_filenames = fnames
            baseline_per_fold_f1 = per_fold

    # Sanity: all configs share the same labels / filenames (LPO is deterministic).
    for tag, info in config_results.items():
        if not np.array_equal(info["labels"], baseline_labels):
            raise RuntimeError(f"{tag}: labels mismatch vs baseline")
        if info["filenames"] != baseline_filenames:
            raise RuntimeError(f"{tag}: filename order differs from baseline")

    # Alias "8-way-diverse" to "8-way-naive" (identical member sets per plan §11.3).
    # Use copy.deepcopy so future mutation of one entry's nested containers
    # (members list, per_class_f1 dict) does not leak into the alias.
    config_results["8-way-diverse"] = copy.deepcopy(config_results["8-way-naive"])
    config_results["8-way-diverse"]["aliased_from"] = "8-way-naive"

    # --- Paired bootstrap each non-baseline config vs 7-way -------------------
    click.echo("\n=== Paired bootstrap (config vs 7-way baseline) ===")
    base_preds = config_results["7-way (baseline)"]["preds"]
    bootstrap_results: dict[str, dict] = {}
    # Skip the alias (8-way-diverse) — bootstrap uses preds, which are identical
    # for naive and diverse, so we compute once and copy below.
    for tag, info in config_results.items():
        if tag == "7-way (baseline)" or info.get("aliased_from"):
            continue
        boot = paired_bootstrap_f1_diff(
            base_preds, info["preds"], baseline_labels,
            n_iter=bootstrap_iter, seed=bootstrap_seed,
        )
        bootstrap_results[tag] = boot
        click.echo(
            f"  {tag:<22}: ΔF1 mean = {boot['mean_diff_pp']:+.3f}pp, "
            f"95% CI [{boot['ci_2.5_pp']:+.3f}, {boot['ci_97.5_pp']:+.3f}]"
        )
    # Copy aliased bootstrap from the source. The bootstrap result dict
    # currently contains only flat float / int values (mean / ci / n_iter /
    # seed), so a shallow `dict(...)` copy is safe; we use copy.deepcopy
    # defensively in case future versions add nested values.
    if "8-way-diverse" in config_results and "8-way-diverse" not in bootstrap_results:
        bootstrap_results["8-way-diverse"] = copy.deepcopy(bootstrap_results["8-way-naive"])
        click.echo(
            f"  {'8-way-diverse':<22}: (aliased from 8-way-naive — identical member set)"
        )

    # --- exp069 standalone (single-member F1) ---------------------------------
    click.echo("\n=== exp069 standalone Macro-F1 ===")
    standalone = load_member_oof(env.artifacts_dir, EXP069_A04_40, num_folds=num_folds)
    standalone_logits = np.concatenate([m[0] for m in standalone], axis=0)
    standalone_labels = np.concatenate([m[1] for m in standalone], axis=0)
    standalone_preds = standalone_logits.argmax(axis=1)
    standalone_f1 = macro_f1(standalone_preds, standalone_labels)
    standalone_per_fold = [
        macro_f1(m[0].argmax(axis=1), m[1]) for m in standalone
    ]
    # Plan §5.2 references the per-fold MEAN F1 (29.87% for a01@40); the
    # concatenated all-sample F1 is a different statistic. We report both.
    standalone_per_fold_mean = float(np.mean(standalone_per_fold))
    click.echo(
        f"  exp069 a04@40 (a01@40=0.2987 per-fold-mean reference): "
        f"concatenated F1 = {standalone_f1*100:.3f}%, "
        f"per-fold F1 mean = {standalone_per_fold_mean*100:.3f}% ± "
        f"{np.std(standalone_per_fold, ddof=1)*100:.3f}"
    )

    # --- Apply Go criteria ------------------------------------------
    # A: standalone per-fold mean F1 ≥ 29.5% (a01@40 = 29.87% per-fold mean)
    # B: 8-way ensemble matrix で +0.20pt F1 with 95% CI lower > -0.10pp
    # C: 全 8-way config が 7-way 以下 (No-Go)
    click.echo("\n=== Go criteria ===")
    criterion_a = standalone_per_fold_mean >= 0.295
    click.echo(
        f"  Criterion A (standalone per-fold mean F1 ≥ 29.5%): "
        f"{'✓' if criterion_a else '✗'} ({standalone_per_fold_mean*100:.3f}%)"
    )

    best_8way_tag = max(
        ["8-way-naive", "8-way-swap", "8-way-diverse", "9-way"],
        key=lambda t: config_results[t]["overall_f1"],
    )
    best_8way_delta_pp = (
        config_results[best_8way_tag]["overall_f1"]
        - config_results["7-way (baseline)"]["overall_f1"]
    ) * 100.0
    best_8way_ci_lower = bootstrap_results[best_8way_tag]["ci_2.5_pp"]
    criterion_b = (best_8way_delta_pp >= 0.20) and (best_8way_ci_lower > -0.10)
    click.echo(
        f"  Criterion B (best ensemble Δ≥+0.20pp AND CI lower > -0.10pp): "
        f"{'✓' if criterion_b else '✗'} "
        f"(best={best_8way_tag}, Δ={best_8way_delta_pp:+.3f}pp, "
        f"CI lower={best_8way_ci_lower:+.3f}pp)"
    )

    all_below_baseline = all(
        config_results[t]["overall_f1"]
        <= config_results["7-way (baseline)"]["overall_f1"]
        for t in ("8-way-naive", "8-way-swap", "8-way-diverse", "9-way")
    )
    criterion_c_no_go = all_below_baseline
    click.echo(
        f"  Criterion C (all 8/9-way configs ≤ 7-way baseline → NO-GO): "
        f"{'✓' if criterion_c_no_go else '✗'}"
    )

    verdict = "GO" if (criterion_a or criterion_b) else "NO-GO"
    click.echo(f"\n=== Overall verdict: {verdict} ===")

    # --- Persist results ------------------------------------------------------
    serialised: dict[str, dict] = {}
    for tag, info in config_results.items():
        serialised[tag] = {
            k: v for k, v in info.items()
            if k not in ("logits", "labels", "filenames", "preds")
        }
    summary = {
        "config": {
            "num_folds": int(num_folds),
            "bootstrap_iter": int(bootstrap_iter),
            "bootstrap_seed": int(bootstrap_seed),
            "production_7way_members": PRODUCTION_7WAY,
            "exp069_member": EXP069_A04_40,
            "exp020_a00_member": EXP020_A00,
        },
        "input_oof_hashes_sha256_16": input_hashes,
        "configs": serialised,
        "bootstrap": bootstrap_results,
        "standalone_exp069": {
            "concatenated_f1": float(standalone_f1),
            "per_fold_f1": list(map(float, standalone_per_fold)),
            "per_fold_f1_mean": standalone_per_fold_mean,
            "per_fold_f1_std": float(np.std(standalone_per_fold, ddof=1)),
            "a01_per_fold_mean_reference": 0.29871,
            "a01_concatenated_f1_reference": 0.30167,
        },
        "go_criteria": {
            "criterion_A_standalone_above_29.5pct": bool(criterion_a),
            "criterion_B_best_8way_plus_0.20pp": bool(criterion_b),
            "criterion_C_no_go_all_below_baseline": bool(criterion_c_no_go),
        },
        "best_8way_tag": best_8way_tag,
        "best_8way_delta_pp": float(best_8way_delta_pp),
        "verdict": verdict,
    }
    json_path = out / "matrix_results.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    click.echo(f"\nWrote {json_path}")


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
