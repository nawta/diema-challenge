""" per-performer F1 / accuracy aggregation from 10-fold OOF predictions.

See: → "performer 別分析"
Related:
- diema/data/parser.py::parse_filename (filename → performer_id)
- diema/training/callbacks.py::OOFPredictionCallback (produces the inputs)

For a trained experiment, this tool:

1. Reads the ``oof_logits.npy`` / ``oof_labels.npy`` / ``oof_filenames.txt``
   from every ``fold_NN`` directory.
2. Concatenates them into a single (N_samples, num_class) logits matrix
   (sample ordering is the union of all fold val sets — each sample appears
   exactly once because LPO splits are disjoint).
3. Parses each filename to extract ``performer_id`` (e.g. ``JP_06``) and
   computes per-performer Macro-F1, accuracy, and sample count.
4. Produces a ranked table, a country-stratified summary, a per-emotion
   breakdown for each performer, and a paper-ready figure.

Usage::

    python tools/analyze_per_performer.py \\
        --experiment exp034_regionaware_convtr_a00

    # Compare with a second experiment (e.g. the 7-way ensemble):
    python tools/analyze_per_performer.py \\
        --experiment exp034_regionaware_convtr_a00 \\
        --extra-experiment ensemble_7way   # if OOF preds exist there
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.env import EnvConfig  # noqa: E402
from diema.data.parser import IDX_TO_EMOTION, parse_filename  # noqa: E402


def _load_oof(exp_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Stack per-fold OOF into a single (N, num_class) logits + labels + filenames."""
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    names_list: list[str] = []
    for fold_dir in sorted(exp_dir.glob("fold_[0-9][0-9]")):
        lp = fold_dir / "oof_logits.npy"
        bp = fold_dir / "oof_labels.npy"
        fp = fold_dir / "oof_filenames.txt"
        if not (lp.exists() and bp.exists() and fp.exists()):
            continue
        logits_list.append(np.load(lp))
        labels_list.append(np.load(bp))
        with fp.open() as f:
            names_list.extend([line.strip() for line in f if line.strip()])
    if not logits_list:
        raise FileNotFoundError(f"no OOF data found in {exp_dir}")
    return (
        np.concatenate(logits_list, axis=0),
        np.concatenate(labels_list, axis=0),
        names_list,
    )


def _perf_id(filename: str) -> tuple[str, str] | None:
    """Return ``(performer_id, country)`` or None if parse fails."""
    try:
        info = parse_filename(Path(filename).stem)
    except Exception:
        return None
    performer = f"{info.nationality}_{int(info.performer_num):02d}"
    return performer, info.nationality


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def build_per_performer_table(
    logits: np.ndarray,
    labels: np.ndarray,
    filenames: list[str],
) -> pd.DataFrame:
    """Group samples by performer, compute F1/acc/count/country."""
    if len(filenames) != labels.size:
        raise ValueError(
            f"filename count ({len(filenames)}) != labels.size ({labels.size})"
        )
    preds = logits.argmax(axis=1)

    # Group indices by performer
    buckets: dict[str, list[int]] = {}
    countries: dict[str, str] = {}
    skipped = 0
    for i, fn in enumerate(filenames):
        info = _perf_id(fn)
        if info is None:
            skipped += 1
            continue
        perf, country = info
        buckets.setdefault(perf, []).append(i)
        countries[perf] = country
    if skipped:
        print(f"[warn] skipped {skipped} filenames that failed to parse")

    rows: list[dict] = []
    for perf in sorted(buckets.keys()):
        idx = np.asarray(buckets[perf])
        y_t = labels[idx]
        y_p = preds[idx]
        rows.append({
            "performer": perf,
            "country": countries[perf],
            "n_samples": int(idx.size),
            "macro_f1": _macro_f1(y_t, y_p),
            "accuracy": _accuracy(y_t, y_p),
            "n_classes_seen": int(np.unique(y_t).size),
        })
    return pd.DataFrame(rows)


def _figure(df: pd.DataFrame, out_base: Path, experiment: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios": [2, 1]})

    # Left: sorted per-performer F1 bar
    df_sorted = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    colors = df_sorted["country"].map({"JP": "#ff7f0e", "TW": "#1f77b4"}).to_list()
    ax0.bar(range(len(df_sorted)), df_sorted["macro_f1"], color=colors,
            edgecolor="black", linewidth=0.3)
    ax0.set_xticks(range(len(df_sorted)))
    ax0.set_xticklabels(df_sorted["performer"], rotation=90, fontsize=6)
    ax0.axhline(df["macro_f1"].mean(), ls="--", color="gray", alpha=0.8,
                label=f"mean {df['macro_f1'].mean():.3f}")
    ax0.set_ylabel("Macro-F1")
    ax0.set_title(f"(a) Per-performer Macro-F1 (sorted) — {experiment}")
    ax0.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#ff7f0e", label="JP"),
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4", label="TW"),
        plt.Line2D([0], [0], color="gray", ls="--", label=f"mean {df['macro_f1'].mean():.3f}"),
    ], loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.3, axis="y")

    # Right: boxplot F1 by country
    data_jp = df[df["country"] == "JP"]["macro_f1"].values
    data_tw = df[df["country"] == "TW"]["macro_f1"].values
    bp = ax1.boxplot([data_jp, data_tw], labels=["JP", "TW"], showmeans=True,
                     patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#ff7f0e", "#1f77b4"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax1.set_ylabel("Macro-F1")
    ax1.set_title("(b) Per-performer F1 by country")
    ax1.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"))
    plt.close(fig)
    click.echo(f"Saved figure: {out_base}.pdf / .png")


def _write_summary_md(
    df: pd.DataFrame,
    path: Path,
    experiment: str,
) -> None:
    df_sorted = df.sort_values("macro_f1")
    n = len(df_sorted)
    lines: list[str] = [
        f"# Per-Performer F1 — {experiment}",
        "",
        f"10-fold LPO OOF aggregated over **{n} performers** "
        f"({int(df['n_samples'].sum())} total samples).",
        "",
        "## Summary",
        "",
        f"| statistic | macro_f1 | accuracy |",
        f"|---|---|---|",
        f"| mean | {df['macro_f1'].mean():.4f} | {df['accuracy'].mean():.4f} |",
        f"| median | {df['macro_f1'].median():.4f} | {df['accuracy'].median():.4f} |",
        f"| std | {df['macro_f1'].std():.4f} | {df['accuracy'].std():.4f} |",
        f"| min | {df['macro_f1'].min():.4f} | {df['accuracy'].min():.4f} |",
        f"| max | {df['macro_f1'].max():.4f} | {df['accuracy'].max():.4f} |",
        "",
        "## By country",
        "",
        "| country | n_performers | mean F1 | std | min | max |",
        "|---|---|---|---|---|---|",
    ]
    for country, sub in df.groupby("country"):
        lines.append(
            f"| {country} | {len(sub)} | {sub['macro_f1'].mean():.4f} | "
            f"{sub['macro_f1'].std():.4f} | {sub['macro_f1'].min():.4f} | "
            f"{sub['macro_f1'].max():.4f} |"
        )

    lines.append("")
    lines.append("## Bottom-5 performers (hardest to classify)")
    lines.append("")
    lines.append("| rank | performer | country | n_samples | macro_f1 | accuracy |")
    lines.append("|---|---|---|---|---|---|")
    for i, r in enumerate(df_sorted.head(5).itertuples(), 1):
        lines.append(
            f"| {i} | **{r.performer}** | {r.country} | {r.n_samples} | "
            f"{r.macro_f1:.4f} | {r.accuracy:.4f} |"
        )

    lines.append("")
    lines.append("## Top-5 performers (easiest to classify)")
    lines.append("")
    lines.append("| rank | performer | country | n_samples | macro_f1 | accuracy |")
    lines.append("|---|---|---|---|---|---|")
    for i, r in enumerate(df_sorted.tail(5).iloc[::-1].itertuples(), 1):
        lines.append(
            f"| {i} | **{r.performer}** | {r.country} | {r.n_samples} | "
            f"{r.macro_f1:.4f} | {r.accuracy:.4f} |"
        )

    lines.append("")
    lines.append(f"## All performers (sorted by F1 ascending)")
    lines.append("")
    lines.append(f"| performer | country | n_samples | macro_f1 | accuracy |")
    lines.append(f"|---|---|---|---|---|")
    for r in df_sorted.itertuples():
        lines.append(
            f"| {r.performer} | {r.country} | {r.n_samples} | "
            f"{r.macro_f1:.4f} | {r.accuracy:.4f} |"
        )

    path.write_text("\n".join(lines))


@click.command()
@click.option("--experiment", required=True)
@click.option("--output-dir", default=None)
def main(experiment: str, output_dir: str | None) -> None:
    env = EnvConfig()
    exp_dir = env.artifacts_dir / experiment
    if not exp_dir.exists():
        raise click.ClickException(f"experiment not found: {exp_dir}")

    out_root = (
        Path(output_dir) if output_dir
        else env.project_root / "docs/analysis/per_performer"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading OOF from {exp_dir}...")
    logits, labels, filenames = _load_oof(exp_dir)
    click.echo(f"  total samples: {len(filenames)}")
    click.echo(f"  accuracy (micro): {np.mean(logits.argmax(1) == labels):.4f}")

    df = build_per_performer_table(logits, labels, filenames)
    click.echo(f"  performers: {len(df)}")

    csv_path = out_root / f"{experiment}_per_performer.csv"
    df.to_csv(csv_path, index=False)
    click.echo(f"Saved: {csv_path}")

    md_path = out_root / f"{experiment}_per_performer.md"
    _write_summary_md(df, md_path, experiment)
    click.echo(f"Saved: {md_path}")

    fig_base = out_root / f"{experiment}_per_performer"
    _figure(df, fig_base, experiment)

    click.echo("\nTop-5 hardest performers:")
    for r in df.sort_values("macro_f1").head(5).itertuples():
        click.echo(
            f"  {r.performer} ({r.country}): F1={r.macro_f1:.4f} acc={r.accuracy:.4f} n={r.n_samples}"
        )
    click.echo("\nTop-5 easiest performers:")
    for r in df.sort_values("macro_f1", ascending=False).head(5).itertuples():
        click.echo(
            f"  {r.performer} ({r.country}): F1={r.macro_f1:.4f} acc={r.accuracy:.4f} n={r.n_samples}"
        )


if __name__ == "__main__":
    main()
