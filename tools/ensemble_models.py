"""Ensemble OOF logits from multiple experiments.

Concatenates the per-fold OOF predictions (saved by OOFPredictionCallback
or tools/recompute_val_f1.py) of each model, aligns them by filename,
and averages the logits to produce a combined CV score.

Usage:
    python tools/ensemble_models.py --experiments exp002_a04_smooth exp003_ctrgcn_a00
"""

import json
from pathlib import Path

import click
import numpy as np

from utils.env import EnvConfig


def load_experiment_oof(exp_dir: Path, num_folds: int = 10):
    """Load concatenated OOF logits, labels, and filenames for an experiment."""
    all_logits = []
    all_labels = []
    all_filenames = []
    for fold in range(num_folds):
        fold_dir = exp_dir / f"fold_{fold:02d}"
        logit_path = fold_dir / "oof_logits.npy"
        label_path = fold_dir / "oof_labels.npy"
        fname_path = fold_dir / "oof_filenames.txt"
        if not logit_path.exists():
            click.echo(f"  WARN: {exp_dir.name} missing fold_{fold:02d} OOF", err=True)
            continue
        all_logits.append(np.load(logit_path))
        all_labels.append(np.load(label_path))
        with open(fname_path) as f:
            all_filenames.extend(f.read().strip().split("\n"))
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return logits, labels, all_filenames


def _per_class_f1(preds: np.ndarray, labels: np.ndarray, num_class: int = 12) -> np.ndarray:
    f1s = np.zeros(num_class, dtype=np.float32)
    for c in range(num_class):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s[c] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1s


def _softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=-1, keepdims=True)


@click.command()
@click.option(
    "--experiments",
    required=True,
    multiple=True,
    help="Experiment folder names (e.g., exp002_a04_smooth). Pass multiple with repeated --experiments.",
)
@click.option(
    "--weights",
    type=str,
    default=None,
    help="Comma-separated ensemble weights (default: equal)",
)
@click.option("--method", type=click.Choice(["logit_mean", "softmax_mean"]), default="logit_mean")
def main(experiments: tuple, weights: str, method: str):
    env = EnvConfig()
    exp_list = list(experiments)

    if weights:
        w_list = [float(x) for x in weights.split(",")]
        assert len(w_list) == len(exp_list), "weight count must match experiment count"
    else:
        w_list = [1.0 / len(exp_list)] * len(exp_list)

    click.echo(f"Ensembling {len(exp_list)} experiments with {method}, weights={w_list}")

    # Load all experiments
    loaded = []
    for exp in exp_list:
        exp_dir = env.artifacts_dir / exp
        logits, labels, filenames = load_experiment_oof(exp_dir)
        click.echo(f"  {exp}: {logits.shape} logits, {len(filenames)} filenames")
        loaded.append((logits, labels, filenames))

    # Align by filename (sort by filename to ensure consistency)
    ref_fnames = loaded[0][2]
    ref_labels = loaded[0][1]
    ref_idx = {fn: i for i, fn in enumerate(ref_fnames)}

    aligned_logits = [loaded[0][0]]
    for logits, labels, filenames in loaded[1:]:
        perm = np.array([ref_idx[fn] for fn in filenames if fn in ref_idx])
        assert len(perm) == len(filenames), "filename mismatch"
        # Invert: reorder logits so they match ref filename order
        inv_perm = np.argsort(perm)
        # logits[inv_perm] gives logits ordered to match ref_fnames[0..N]
        # But we want: aligned[i] = logits of fn=ref_fnames[i]
        # Create mapping: for each ref position, find which logits row has that filename
        fn_to_row = {fn: i for i, fn in enumerate(filenames)}
        row_order = [fn_to_row[fn] for fn in ref_fnames]
        aligned_logits.append(logits[row_order])
        assert np.array_equal(labels[row_order], ref_labels), "label mismatch after alignment"

    # Combine
    if method == "logit_mean":
        combined = sum(w * L for w, L in zip(w_list, aligned_logits))
    else:  # softmax_mean
        combined = sum(w * _softmax(L) for w, L in zip(w_list, aligned_logits))

    preds = combined.argmax(axis=1)
    acc = float((preds == ref_labels).mean())
    f1s = _per_class_f1(preds, ref_labels)
    macro_f1 = float(f1s.mean())

    click.echo(f"\nEnsemble results:")
    click.echo(f"  Accuracy: {acc*100:.2f}%")
    click.echo(f"  Macro-F1: {macro_f1*100:.2f}%")
    click.echo(f"  Per-class F1: " + " ".join(f"{x*100:.1f}" for x in f1s))

    # Individual results for comparison
    click.echo("\nIndividual:")
    for name, L in zip(exp_list, aligned_logits):
        p = L.argmax(axis=1)
        a = float((p == ref_labels).mean())
        f = float(_per_class_f1(p, ref_labels).mean())
        click.echo(f"  {name}: Acc {a*100:.2f}%, F1 {f*100:.2f}%")


if __name__ == "__main__":
    main()
