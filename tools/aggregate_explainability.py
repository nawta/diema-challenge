""" aggregate explainability results into a single report.

Related:
- tools/explain_part_masking.py / explain_temporal_masking.py / explain_stability.py
- experiments/exp058_explainability_eval
- tools/generate_paper_figures.py

Reads the four per-analysis output directories and produces
``docs/analysis/explainability_report.md`` — a compact summary that a
reviewer or paper writer can copy directly into the manuscript.

Usage::

    python tools/aggregate_explainability.py \
        --experiment exp034_regionaware_convtr_a00

    # Or compare multiple experiments side by side
    python tools/aggregate_explainability.py \
        --experiment exp020_conv1d_transformer_a01 \
        --experiment exp034_regionaware_convtr_a00
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.env import EnvConfig  # noqa: E402


def _load_part_masking_summary(pm_dir: Path, exp: str) -> dict:
    csv_path = pm_dir / f"{exp}_part_importance.csv"
    if not csv_path.exists():
        return {"status": "missing", "path": str(csv_path)}
    df = pd.read_csv(csv_path)
    by_part = df.groupby("part")["f1_drop"].agg(["mean", "std"]).round(4)
    base_f1 = df["baseline_f1"].mean()

    faithfulness_files = sorted(pm_dir.glob(f"{exp}_fold[0-9][0-9]_faithfulness.json"))
    aucs = {"important_first": [], "reverse": [], "random": []}
    for p in faithfulness_files:
        d = json.loads(p.read_text())
        for k in aucs:
            aucs[k].append(d["aucs"][k])
    auc_mean = {k: float(np.mean(v)) if v else None for k, v in aucs.items()}
    auc_std = {k: float(np.std(v)) if v else None for k, v in aucs.items()}

    return {
        "status": "ok",
        "baseline_f1": float(base_f1),
        "num_folds": len(faithfulness_files),
        "part_drops_mean": by_part["mean"].to_dict(),
        "part_drops_std": by_part["std"].to_dict(),
        "auc_mean": auc_mean,
        "auc_std": auc_std,
    }


def _load_temporal_masking_summary(tm_dir: Path, exp: str) -> dict:
    csv_path = tm_dir / f"{exp}_temporal_importance.csv"
    if not csv_path.exists():
        return {"status": "missing", "path": str(csv_path)}
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {"status": "empty"}
    out = {
        "status": "ok",
        "num_folds": len(df),
        "num_windows": int(df["num_windows"].iloc[0]),
        "baseline_f1": float(df["baseline_f1"].mean()),
    }
    for col in ("auc_salient_first", "auc_reverse", "auc_random"):
        if col in df.columns:
            out[f"{col}_mean"] = float(df[col].mean())
            out[f"{col}_std"] = float(df[col].std())
    return out


def _load_stability_summary(st_dir: Path, exp: str) -> dict:
    csv_path = st_dir / f"{exp}_stability.csv"
    if not csv_path.exists():
        return {"status": "missing", "path": str(csv_path)}
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {"status": "empty"}
    # Mean Spearman per sigma
    rho_by_sigma = df.groupby("sigma")["spearman_vs_ref"].agg(
        ["mean", "std"]).round(3)
    f1_by_sigma = df.groupby("sigma")["baseline_f1"].mean().round(4)
    return {
        "status": "ok",
        "num_folds": int(df[df["sigma"] == 0.0].shape[0]),
        "sigmas": sorted(df["sigma"].unique().tolist()),
        "spearman_mean": rho_by_sigma["mean"].to_dict(),
        "spearman_std": rho_by_sigma["std"].to_dict(),
        "baseline_f1_by_sigma": f1_by_sigma.to_dict(),
    }


def _load_cross_cultural_summary(pm_dir: Path, exp: str) -> dict:
    jp_csv = pm_dir / f"{exp}_part_importance_JP.csv"
    tw_csv = pm_dir / f"{exp}_part_importance_TW.csv"
    if not jp_csv.exists() or not tw_csv.exists():
        return {
            "status": "missing",
            "jp_present": jp_csv.exists(),
            "tw_present": tw_csv.exists(),
        }
    jp = pd.read_csv(jp_csv).groupby("part")["f1_drop"].mean()
    tw = pd.read_csv(tw_csv).groupby("part")["f1_drop"].mean()
    diff = (jp - tw).reindex(sorted(set(jp.index) | set(tw.index))).round(4)
    return {
        "status": "ok",
        "jp_drops": jp.round(4).to_dict(),
        "tw_drops": tw.round(4).to_dict(),
        "jp_minus_tw_drop": diff.to_dict(),
    }


def _render_report(
    output_path: Path,
    experiments: list[str],
    summaries: list[dict],
) -> None:
    lines: list[str] = [
        "# Explainability Report ",
        "",
        f"**Generated**: aggregate of `tools/explain_part_masking.py`, "
        f"`tools/explain_temporal_masking.py`, `tools/explain_stability.py` "
        f"outputs over the following experiments:",
        "",
    ]
    for exp in experiments:
        lines.append(f"- `{exp}`")
    lines.extend([
        "",
        "## 1. Part-masking faithfulness ",
        "",
        "| Experiment | Baseline F1 | AUC important_first | AUC random | AUC reverse | Gap |",
        "|---|---|---|---|---|---|",
    ])
    for exp, s in zip(experiments, summaries):
        pm = s["part_masking"]
        if pm["status"] != "ok":
            lines.append(f"| `{exp}` | — | (missing) | — | — | — |")
            continue
        am = pm["auc_mean"]
        as_ = pm["auc_std"]
        gap = am["important_first"] - am["reverse"]
        lines.append(
            f"| `{exp}` "
            f"| {pm['baseline_f1']:.4f} "
            f"| {am['important_first']:.3f} ± {as_['important_first']:.3f} "
            f"| {am['random']:.3f} ± {as_['random']:.3f} "
            f"| {am['reverse']:.3f} ± {as_['reverse']:.3f} "
            f"| {gap:+.3f} |"
        )
    lines.extend([
        "",
        "**Interpretation**: `important_first > random > reverse` gap measures how"
        " faithful the ranking is. Larger gap = more informative explanation.",
        "",
        "## 2. Temporal-masking faithfulness ",
        "",
        "| Experiment | Baseline F1 | Windows | AUC salient_first | AUC random | AUC reverse |",
        "|---|---|---|---|---|---|",
    ])
    for exp, s in zip(experiments, summaries):
        tm = s["temporal_masking"]
        if tm["status"] != "ok":
            lines.append(f"| `{exp}` | — | — | (missing) | — | — |")
            continue
        lines.append(
            f"| `{exp}` "
            f"| {tm['baseline_f1']:.4f} "
            f"| {tm['num_windows']} "
            f"| {tm['auc_salient_first_mean']:.3f} ± {tm['auc_salient_first_std']:.3f} "
            f"| {tm['auc_random_mean']:.3f} ± {tm['auc_random_std']:.3f} "
            f"| {tm['auc_reverse_mean']:.3f} ± {tm['auc_reverse_std']:.3f} |"
        )
    lines.extend([
        "",
        "**Note**: gradient-based saliency only; rationale-driven temporal"
        " saliency (Gemini) will populate a second table when the"
        " G-rule gate opens.",
        "",
        "## 3. Explanation stability under input noise ",
        "",
        "Spearman rank correlation of the part-importance ranking vs σ=0"
        " (mean ± std across folds):",
        "",
        "| Experiment | σ=0.005 | σ=0.010 | σ=0.020 |",
        "|---|---|---|---|",
    ])
    for exp, s in zip(experiments, summaries):
        st = s["stability"]
        if st["status"] != "ok":
            lines.append(f"| `{exp}` | (missing) | (missing) | (missing) |")
            continue
        cells: list[str] = []
        for sigma in (0.005, 0.010, 0.020):
            m = st["spearman_mean"].get(sigma)
            sd = st["spearman_std"].get(sigma)
            if m is None:
                cells.append("—")
            else:
                cells.append(f"{m:+.3f} ± {sd:.3f}")
        lines.append(f"| `{exp}` | {' | '.join(cells)} |")
    lines.extend([
        "",
        "**Interpretation**: Spearman close to +1 → explanation ranking is"
        " stable under small input perturbations; close to 0 → noise collapses"
        " the explanation.",
        "",
        "## 4. Cross-cultural (JP vs TW) — ",
        "",
        "| Experiment | Head JP | Head TW | Head diff | Largest |JP-TW| part |",
        "|---|---|---|---|---|",
    ])
    for exp, s in zip(experiments, summaries):
        cc = s["cross_cultural"]
        if cc["status"] != "ok":
            lines.append(f"| `{exp}` | (missing) | (missing) | — | — |")
            continue
        jp_h = cc["jp_drops"].get("head")
        tw_h = cc["tw_drops"].get("head")
        diff = cc["jp_minus_tw_drop"]
        top = max(diff, key=lambda k: abs(diff[k])) if diff else "—"
        lines.append(
            f"| `{exp}` "
            f"| {jp_h:+.3f} | {tw_h:+.3f} "
            f"| {(jp_h - tw_h):+.3f} | {top} ({diff.get(top):+.3f}) |"
        )
    lines.extend([
        "",
        "---",
        "",
        "_Regenerate with_ `python tools/aggregate_explainability.py --experiment ...`",
        "",
    ])
    output_path.write_text("\n".join(lines))


@click.command()
@click.option("--experiment", "experiments", required=True, multiple=True)
@click.option("--output", default="docs/analysis/explainability_report.md")
@click.option("--part-masking-dir", default="docs/analysis/part_masking")
@click.option("--temporal-masking-dir", default="docs/analysis/temporal_masking")
@click.option("--stability-dir", default="docs/analysis/stability")
def main(experiments: tuple[str, ...], output: str,
         part_masking_dir: str, temporal_masking_dir: str,
         stability_dir: str) -> None:
    env = EnvConfig()
    pm_dir = env.project_root / part_masking_dir
    tm_dir = env.project_root / temporal_masking_dir
    st_dir = env.project_root / stability_dir
    output_path = env.project_root / output

    summaries: list[dict] = []
    for exp in experiments:
        summaries.append({
            "part_masking": _load_part_masking_summary(pm_dir, exp),
            "temporal_masking": _load_temporal_masking_summary(tm_dir, exp),
            "stability": _load_stability_summary(st_dir, exp),
            "cross_cultural": _load_cross_cultural_summary(pm_dir, exp),
        })
    _render_report(output_path, list(experiments), summaries)
    click.echo(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
