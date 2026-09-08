""" diagnostic analysis: confusion matrix, per-class F1, per-performer accuracy.

Compares exp001 (baseline) and exp002_a04_smooth (label_smoothing variant).
Reads OOF logits/labels saved by tools/recompute_val_f1.py.

Outputs to docs/diagnosis_phase3a.md.

Usage:
    python tools/diagnose_baseline.py
"""

import json
from pathlib import Path

import click
import numpy as np

from utils.env import EnvConfig
from diema.data.parser import EMOTION_TO_IDX, parse_filename


IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}


def aggregate_oof(exp_dir: Path, num_folds: int = 10):
    """Concatenate per-fold OOF logits and labels into a single array."""
    all_logits, all_labels, all_filenames = [], [], []
    for fold in range(num_folds):
        fold_dir = exp_dir / f"fold_{fold:02d}"
        logit_path = fold_dir / "oof_logits.npy"
        label_path = fold_dir / "oof_labels.npy"
        fname_path = fold_dir / "oof_filenames.txt"
        if not logit_path.exists():
            click.echo(f"  WARN: missing OOF for fold_{fold:02d}", err=True)
            continue
        all_logits.append(np.load(logit_path))
        all_labels.append(np.load(label_path))
        with open(fname_path) as f:
            all_filenames.extend(f.read().strip().split("\n"))
    return np.concatenate(all_logits), np.concatenate(all_labels), all_filenames


def compute_confusion(preds: np.ndarray, labels: np.ndarray, num_class: int = 12):
    cm = np.zeros((num_class, num_class), dtype=np.int64)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    return cm


def per_class_f1(cm: np.ndarray):
    num_class = cm.shape[0]
    f1s = []
    for c in range(num_class):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return np.array(f1s)


def per_performer_acc(preds, labels, filenames):
    """Group predictions by performer (parsed from filename)."""
    from collections import defaultdict
    by_perf = defaultdict(lambda: {"correct": 0, "total": 0})
    for p, l, fn in zip(preds, labels, filenames):
        try:
            info = parse_filename(fn)
            perf = info.performer_id  # already "JP_06" / "TW_30" format
        except Exception:
            perf = "unknown"
        by_perf[perf]["total"] += 1
        if p == l:
            by_perf[perf]["correct"] += 1
    return {perf: v["correct"] / v["total"] for perf, v in by_perf.items() if v["total"] > 0}


def format_confusion_md(cm: np.ndarray, class_names: list[str]) -> str:
    lines = ["| true \\ pred | " + " | ".join(c[:4] for c in class_names) + " |"]
    lines.append("|" + "---|" * (len(class_names) + 1))
    for i, name in enumerate(class_names):
        row = [f"**{name[:4]}**"] + [str(cm[i, j]) for j in range(len(class_names))]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


@click.command()
@click.option("--num-class", type=int, default=12)
def main(num_class: int):
    env = EnvConfig()
    class_names = [IDX_TO_EMOTION[i] for i in range(num_class)]

    out_lines = ["# 診断分析: exp001 vs exp002 a04 (label_smoothing=0.1)\n"]

    experiments = [
        ("exp001_benchmark_repro", "baseline"),
        ("exp002_a04_smooth",      "label_smoothing=0.1"),
    ]

    summaries = {}
    for exp_name, label in experiments:
        exp_dir = env.artifacts_dir / exp_name
        click.echo(f"\nProcessing {exp_name}...")
        logits, labels, filenames = aggregate_oof(exp_dir)
        preds = logits.argmax(axis=1)
        n = len(labels)
        cm = compute_confusion(preds, labels, num_class)
        f1s = per_class_f1(cm)
        macro_f1 = float(f1s.mean())
        acc = float((preds == labels).mean())
        perf_accs = per_performer_acc(preds, labels, filenames)
        perf_arr = np.array(list(perf_accs.values()))

        summaries[exp_name] = {
            "label": label,
            "n": n,
            "acc": acc,
            "macro_f1": macro_f1,
            "per_class_f1": f1s,
            "cm": cm,
            "perf_accs": perf_accs,
            "perf_mean": float(perf_arr.mean()),
            "perf_std":  float(perf_arr.std()),
            "perf_min":  float(perf_arr.min()),
            "perf_max":  float(perf_arr.max()),
        }

        out_lines.append(f"## {exp_name} ({label})\n")
        out_lines.append(f"- N val samples (10-fold concat): {n}")
        out_lines.append(f"- Overall accuracy: {acc*100:.2f}%")
        out_lines.append(f"- Macro-F1: {macro_f1*100:.2f}%\n")
        out_lines.append("### Per-class F1\n")
        out_lines.append("| Class | F1 (%) |")
        out_lines.append("|---|---|")
        for i, name in enumerate(class_names):
            out_lines.append(f"| {name} | {f1s[i]*100:.2f} |")
        out_lines.append("")
        out_lines.append("### Confusion matrix (rows=true, cols=pred)\n")
        out_lines.append(format_confusion_md(cm, class_names))
        out_lines.append("\n### Per-performer accuracy\n")
        out_lines.append(f"- mean: {perf_arr.mean()*100:.2f}%")
        out_lines.append(f"- std:  {perf_arr.std()*100:.2f}%")
        out_lines.append(f"- min:  {perf_arr.min()*100:.2f}% (worst performer)")
        out_lines.append(f"- max:  {perf_arr.max()*100:.2f}% (best performer)")
        worst5 = sorted(perf_accs.items(), key=lambda x: x[1])[:5]
        best5 = sorted(perf_accs.items(), key=lambda x: -x[1])[:5]
        out_lines.append("\n#### worst 5 performers")
        for p, a in worst5:
            out_lines.append(f"- {p}: {a*100:.2f}%")
        out_lines.append("\n#### best 5 performers")
        for p, a in best5:
            out_lines.append(f"- {p}: {a*100:.2f}%")
        out_lines.append("\n---\n")

    # Comparison
    if len(summaries) == 2:
        e1 = summaries["exp001_benchmark_repro"]
        e2 = summaries["exp002_a04_smooth"]
        out_lines.append("## Comparison: exp002 a04 vs exp001 baseline\n")
        out_lines.append(f"- Overall acc: {e1['acc']*100:.2f}% → {e2['acc']*100:.2f}% ({(e2['acc']-e1['acc'])*100:+.2f}pt)")
        out_lines.append(f"- Macro-F1:    {e1['macro_f1']*100:.2f}% → {e2['macro_f1']*100:.2f}% ({(e2['macro_f1']-e1['macro_f1'])*100:+.2f}pt)\n")
        out_lines.append("### Per-class F1 delta\n")
        out_lines.append("| Class | exp001 | exp002 a04 | Δ |")
        out_lines.append("|---|---|---|---|")
        f1_deltas = []
        for i, name in enumerate(class_names):
            d = (e2["per_class_f1"][i] - e1["per_class_f1"][i]) * 100
            f1_deltas.append(d)
            out_lines.append(f"| {name} | {e1['per_class_f1'][i]*100:.2f} | {e2['per_class_f1'][i]*100:.2f} | {d:+.2f} |")
        improved = sum(1 for d in f1_deltas if d > 0)
        out_lines.append(f"\nClasses improved: {improved}/{len(class_names)}")

        out_lines.append("\n### Per-performer accuracy delta\n")
        common = set(e1["perf_accs"]) & set(e2["perf_accs"])
        deltas = [(e2["perf_accs"][p] - e1["perf_accs"][p]) * 100 for p in common]
        out_lines.append(f"- mean delta: {np.mean(deltas):+.2f}pt")
        out_lines.append(f"- std  delta: {np.std(deltas):.2f}pt")
        out_lines.append(f"- improved performers: {sum(1 for d in deltas if d > 0)}/{len(deltas)}")

    out_path = Path("docs/diagnosis_phase3a.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
    click.echo(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
