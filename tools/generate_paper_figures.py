""" generate publication-quality figures for the paper.

Related:
- tools/explain_part_masking.py / explain_temporal_masking.py / explain_stability.py

Renders (high-DPI PDF + PNG) the Figures 4-6 + A1-A3 for the system
description paper, using the already-produced CSV/JSON analysis outputs.
Re-runs cheaply — no model inference; just plots.

Usage::

    python tools/generate_paper_figures.py --experiment exp034_regionaware_convtr_a00
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
from diema.models.skeleton_graph import DIEMA_BODY_PARTS  # noqa: E402

PARTS_ORDERED = list(DIEMA_BODY_PARTS.keys())
CANONICAL_EMOTIONS = [
    "anger", "contempt", "disgust", "fear", "gratitude", "guilt",
    "jealousy", "joy", "pride", "sadness", "shame", "surprise",
]


def _setup_matplotlib() -> object:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,  # embed TrueType
        "ps.fonttype": 42,
    })
    return plt


def figure_part_importance_heatmap(csv_path: Path, out_base: Path) -> None:
    """Figure 4: 12-emotion × 6-part F1 drop heatmap (10-fold mean)."""
    plt = _setup_matplotlib()
    df = pd.read_csv(csv_path)
    agg = df.groupby("part").mean(numeric_only=True)
    drop_cols = [f"f1_drop_{e}" for e in CANONICAL_EMOTIONS]
    missing = [c for c in drop_cols if c not in agg.columns]
    if missing:
        click.echo(f"[warn] missing columns: {missing}", err=True)
        return
    heatmap = agg[drop_cols].reindex(PARTS_ORDERED)
    heatmap.columns = CANONICAL_EMOTIONS

    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    im = ax.imshow(heatmap.values, aspect="auto", cmap="RdYlBu_r",
                   vmin=-0.05, vmax=0.35)
    ax.set_xticks(range(len(CANONICAL_EMOTIONS)))
    ax.set_xticklabels(CANONICAL_EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(len(PARTS_ORDERED)))
    ax.set_yticklabels(PARTS_ORDERED)
    ax.set_xlabel("Emotion class")
    ax.set_ylabel("Body part (zero-masked)")
    fig.colorbar(im, ax=ax, label="Macro-F1 drop (baseline − masked)")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"))
    plt.close(fig)
    click.echo(f"Saved Figure 4: {out_base}.pdf / .png")


def figure_faithfulness_curves(json_dir: Path, out_base: Path,
                               experiment: str) -> None:
    """Figure 5: faithfulness curves (three orderings × k=0..6 cumulative mask)."""
    plt = _setup_matplotlib()
    fold_files = sorted(json_dir.glob(f"{experiment}_fold[0-9][0-9]_faithfulness.json"))
    if not fold_files:
        click.echo(f"[warn] no faithfulness JSONs in {json_dir}", err=True)
        return

    # Stack curves: shape (n_folds, n_steps) per ordering
    curves_by_order: dict[str, list[list[float]]] = {
        "important_first": [], "reverse": [], "random": []
    }
    base_f1 = []
    for p in fold_files:
        d = json.loads(p.read_text())
        base_f1.append(d["baseline_f1"])
        for o in curves_by_order:
            curves_by_order[o].append(d["curves"][o])
    mean_base = np.mean(base_f1)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    colors = {"important_first": "C3", "random": "C0", "reverse": "C2"}
    labels = {
        "important_first": "important-first (ours)",
        "random": "random",
        "reverse": "reverse",
    }
    for o in ("important_first", "random", "reverse"):
        arr = np.array(curves_by_order[o])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        k = np.arange(arr.shape[1])
        ax.plot(k, mean, "-o", color=colors[o], label=labels[o], linewidth=2)
        ax.fill_between(k, mean - std, mean + std, color=colors[o], alpha=0.15)
    ax.axhline(mean_base, ls="--", color="gray", alpha=0.5,
               label=f"unmasked baseline {mean_base:.3f}")
    ax.set_xlabel("Number of cumulatively masked body parts (k)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Part-masking faithfulness (10-fold LPO-CV)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"))
    plt.close(fig)
    click.echo(f"Saved Figure 5: {out_base}.pdf / .png")


def figure_cross_cultural(jp_csv: Path, tw_csv: Path, out_base: Path) -> None:
    """Figure 6: JP vs TW per-part F1 drop with diff bar + heatmap of per-class diff."""
    plt = _setup_matplotlib()
    jp = pd.read_csv(jp_csv)
    tw = pd.read_csv(tw_csv)
    jp_mean = jp.groupby("part")["f1_drop"].agg(["mean", "std"]).reindex(PARTS_ORDERED)
    tw_mean = tw.groupby("part")["f1_drop"].agg(["mean", "std"]).reindex(PARTS_ORDERED)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.8),
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    # Left: paired bars
    x = np.arange(len(PARTS_ORDERED))
    w = 0.38
    ax0.bar(x - w / 2, jp_mean["mean"], w, yerr=jp_mean["std"], capsize=3,
            label="JP", color="C1", alpha=0.9)
    ax0.bar(x + w / 2, tw_mean["mean"], w, yerr=tw_mean["std"], capsize=3,
            label="TW", color="C0", alpha=0.9)
    ax0.set_xticks(x)
    ax0.set_xticklabels(PARTS_ORDERED, rotation=30, ha="right")
    ax0.set_ylabel("F1 drop")
    ax0.set_title("(a) Per-part importance by country")
    ax0.grid(True, alpha=0.3, axis="y")
    ax0.legend()

    # Right: per-class diff heatmap (JP - TW)
    jp_c = jp.groupby("part").mean(numeric_only=True)
    tw_c = tw.groupby("part").mean(numeric_only=True)
    drop_cols = [f"f1_drop_{e}" for e in CANONICAL_EMOTIONS]
    if all(c in jp_c.columns for c in drop_cols):
        diff = (jp_c[drop_cols] - tw_c[drop_cols]).reindex(PARTS_ORDERED)
        diff.columns = CANONICAL_EMOTIONS
        vmax = max(abs(diff.values.min()), abs(diff.values.max()), 0.1)
        im = ax1.imshow(diff.values, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax)
        ax1.set_xticks(range(len(CANONICAL_EMOTIONS)))
        ax1.set_xticklabels(CANONICAL_EMOTIONS, rotation=45, ha="right")
        ax1.set_yticks(range(len(PARTS_ORDERED)))
        ax1.set_yticklabels(PARTS_ORDERED)
        ax1.set_title("(b) Per-class F1-drop diff (JP − TW)")
        fig.colorbar(im, ax=ax1)

    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"))
    plt.close(fig)
    click.echo(f"Saved Figure 6: {out_base}.pdf / .png")


def figure_intensity_breakdown(
    csvs: dict[str, Path], out_base: Path, experiment: str,
) -> None:
    """Figure 7: per-part F1 drop stratified by DIEM-A intensity (L/M/H)."""
    plt = _setup_matplotlib()

    means: dict[str, dict[str, float]] = {}
    stds: dict[str, dict[str, float]] = {}
    baselines: dict[str, tuple[float, float]] = {}
    for intensity, path in csvs.items():
        if not path.exists():
            click.echo(f"[warn] intensity {intensity} CSV missing: {path}", err=True)
            continue
        df = pd.read_csv(path)
        by_part = df.groupby("part")["f1_drop"].agg(["mean", "std"]).reindex(PARTS_ORDERED)
        means[intensity] = by_part["mean"].to_dict()
        stds[intensity] = by_part["std"].to_dict()
        # baseline_f1 is the same for every row within a fold, so take a per-fold first value.
        baselines[intensity] = (
            float(df.drop_duplicates("fold")["baseline_f1"].mean()),
            float(df.drop_duplicates("fold")["baseline_f1"].std()),
        )

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2.2, 1]}
    )

    # Left: grouped bar chart — per-part F1 drop × intensity
    colors = {"L": "#8ecae6", "M": "#ffb703", "H": "#fb8500"}
    x = np.arange(len(PARTS_ORDERED))
    w = 0.26
    for i, intensity in enumerate(("L", "M", "H")):
        if intensity not in means:
            continue
        y = [means[intensity][p] for p in PARTS_ORDERED]
        err = [stds[intensity][p] for p in PARTS_ORDERED]
        ax0.bar(
            x + (i - 1) * w, y, w, yerr=err, capsize=3,
            label=f"intensity {intensity}", color=colors[intensity], edgecolor="black",
            linewidth=0.4,
        )
    ax0.set_xticks(x)
    ax0.set_xticklabels(PARTS_ORDERED, rotation=30, ha="right")
    ax0.set_ylabel("Macro-F1 drop")
    ax0.set_title("(a) Per-part importance by DIEM-A intensity tag")
    ax0.grid(True, alpha=0.3, axis="y")
    ax0.legend(loc="upper right", framealpha=0.9)

    # Right: baseline F1 by intensity + faithfulness gap
    ax1.bar(
        list(baselines.keys()),
        [v[0] for v in baselines.values()],
        yerr=[v[1] for v in baselines.values()],
        capsize=4,
        color=[colors[k] for k in baselines.keys()],
        edgecolor="black", linewidth=0.4,
    )
    ax1.set_ylabel("Baseline Macro-F1")
    ax1.set_title("(b) Baseline F1 by intensity")
    ax1.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"))
    plt.close(fig)
    click.echo(f"Saved Figure 7: {out_base}.pdf / .png")


def figure_part_drop_bar(csv_path: Path, out_base: Path) -> None:
    """Appendix figure: mean F1 drop per body part with 10-fold error bars."""
    plt = _setup_matplotlib()
    df = pd.read_csv(csv_path)
    agg = df.groupby("part")["f1_drop"].agg(["mean", "std"]).reindex(PARTS_ORDERED)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(range(len(agg)), agg["mean"], yerr=agg["std"], capsize=4,
           color="C0", alpha=0.85)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(agg.index, rotation=30, ha="right")
    ax.set_ylabel("Macro-F1 drop")
    ax.set_xlabel("Body part (zero-masked)")
    ax.set_title("Per-part importance, 10-fold mean ± std")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"))
    plt.close(fig)
    click.echo(f"Saved part-drop bar: {out_base}.pdf / .png")


@click.command()
@click.option("--experiment", default="exp034_regionaware_convtr_a00",
              help="Experiment to draw figures for")
@click.option("--part-masking-dir", default="docs/analysis/part_masking")
@click.option("--output-dir", default="docs/analysis/paper_figures")
def main(experiment: str, part_masking_dir: str, output_dir: str) -> None:
    env = EnvConfig()
    pm_dir = env.project_root / part_masking_dir
    out_dir = env.project_root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = pm_dir / f"{experiment}_part_importance.csv"
    if csv_path.exists():
        figure_part_importance_heatmap(csv_path, out_dir / f"fig4_part_heatmap_{experiment}")
        figure_part_drop_bar(csv_path, out_dir / f"figA_part_drop_bar_{experiment}")
    else:
        click.echo(f"[skip] {csv_path} not found", err=True)

    figure_faithfulness_curves(pm_dir, out_dir / f"fig5_faithfulness_{experiment}", experiment)

    jp_csv = pm_dir / f"{experiment}_part_importance_JP.csv"
    tw_csv = pm_dir / f"{experiment}_part_importance_TW.csv"
    if jp_csv.exists() and tw_csv.exists():
        figure_cross_cultural(jp_csv, tw_csv, out_dir / f"fig6_cross_cultural_{experiment}")
    else:
        click.echo(f"[skip] cross-cultural CSVs missing", err=True)

    intensity_csvs = {
        "L": pm_dir / f"{experiment}_part_importance_intL.csv",
        "M": pm_dir / f"{experiment}_part_importance_intM.csv",
        "H": pm_dir / f"{experiment}_part_importance_intH.csv",
    }
    if all(p.exists() for p in intensity_csvs.values()):
        figure_intensity_breakdown(
            intensity_csvs, out_dir / f"fig7_intensity_{experiment}", experiment
        )
    else:
        missing = [k for k, p in intensity_csvs.items() if not p.exists()]
        click.echo(f"[skip] intensity CSVs missing: {missing}", err=True)


if __name__ == "__main__":
    main()
