"""exp057: Capture-domain classifier probe.

Goal: quantify how much capture-domain information (country / site) is
leaked by the C3D marker statistics alone. Trains a small MLP classifier
from the 51-D C3D feature vector → country label (JP / TW). High accuracy
means the C3D stats encode subject / site identity — a relevant signal
for the late-fusion model (exp056) but also a failure mode if the model
learns to rely on it.

Implementation is deliberately minimal — no Lightning, no augmentation,
no wandb — because the question is "does a linear/MLP probe trivially
separate the two acquisition sites?" not "what's the best classifier?".

Usage::

    python -m experiments.exp057_capture_domain_probe.run
    # or explicit paths:
    python -m experiments.exp057_capture_domain_probe.run \\
        cache=output/c3d_stats_cache.npz target=country

Reads cached C3D stats from ``output/c3d_stats_cache.npz`` (builds it if
missing via tools/cache_c3d_stats.py). Writes a per-performer LPO accuracy
JSON + a per-class confusion matrix to
``output/artifacts/exp057_capture_domain_probe/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.env import EnvConfig  # noqa: E402
from diema.data.parser import parse_filename  # noqa: E402


def _load_cache(cache_path: Path) -> tuple[np.ndarray, list[str]]:
    data = np.load(cache_path, allow_pickle=False)
    features = data["features"].astype(np.float32)
    filenames = [str(s) for s in data["filenames"].tolist()]
    return features, filenames


def _labels_from_filenames(
    filenames: list[str], target: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Derive labels + performer ids from filename stems.

    Returns: (labels (int), class_names, performer_ids (int))
    - `country`: JP=0, TW=1
    """
    labels: list[int] = []
    perf_ids: list[str] = []
    class_set: list[str] = []
    for fn in filenames:
        try:
            info = parse_filename(Path(fn).stem)
            value = info.nationality if target == "country" else None
        except Exception:
            info = None
            value = None
        if value is None or info is None:
            # unparseable; mark as -1 and skip during training
            labels.append(-1)
            perf_ids.append("")
            continue
        if value not in class_set:
            class_set.append(value)
        labels.append(class_set.index(value))
        perf_ids.append(f"{info.nationality}_{int(info.performer_num):02d}")
    # Sort class_set for stability then remap labels
    order = sorted(class_set)
    remap = {old: order.index(c) for old, c in enumerate(class_set)}
    labels_arr = np.asarray([remap.get(l, -1) for l in labels], dtype=np.int64)
    perf_arr = np.asarray(perf_ids)
    return labels_arr, order, perf_arr


def _standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (features - mean) / std


def _train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    max_iter: int = 500,
) -> tuple[float, np.ndarray]:
    """Scikit-learn LogisticRegression probe. Returns (accuracy, val_pred)."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=max_iter, n_jobs=1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    acc = float(np.mean(pred == y_val))
    return acc, pred


def _leave_one_performer_out(
    features: np.ndarray,
    labels: np.ndarray,
    performer_ids: np.ndarray,
) -> dict:
    """Run LPO CV: for each held-out performer, train on the rest, test on them."""
    valid = labels >= 0
    if not valid.any():
        raise ValueError("no valid labels after filename parsing")
    features = features[valid]
    labels = labels[valid]
    performer_ids = performer_ids[valid]

    uniq_perf = sorted(set(performer_ids))
    click.echo(f"LPO CV: {len(uniq_perf)} performers, "
               f"{len(features)} samples, {labels.max() + 1} classes")

    per_perf_acc: dict[str, float] = {}
    all_true: list[int] = []
    all_pred: list[int] = []
    for perf in uniq_perf:
        val_mask = performer_ids == perf
        train_mask = ~val_mask
        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        X_train_s = _standardize(features[train_mask])
        X_val_s = (features[val_mask] - features[train_mask].mean(axis=0)) / \
                  np.where(features[train_mask].std(axis=0) < 1e-8, 1.0,
                           features[train_mask].std(axis=0))
        acc, pred = _train_logistic_regression(
            X_train_s, labels[train_mask],
            X_val_s, labels[val_mask],
        )
        per_perf_acc[perf] = acc
        all_true.extend(labels[val_mask].tolist())
        all_pred.extend(pred.tolist())

    overall_acc = float(np.mean(np.array(all_pred) == np.array(all_true)))
    return {
        "per_performer_acc": per_perf_acc,
        "overall_acc": overall_acc,
        "y_true": all_true,
        "y_pred": all_pred,
    }


@click.command()
@click.option("--cache", default="output/c3d_stats_cache.npz")
@click.option("--target", default="country",
              type=click.Choice(["country"]),
              help="Classification target (more to come when we need site-id etc.)")
@click.option("--output-dir", default=None)
def main(cache: str, target: str, output_dir: str | None) -> None:
    env = EnvConfig()
    cache_path = Path(cache)
    if not cache_path.is_absolute():
        cache_path = env.project_root / cache_path
    if not cache_path.exists():
        raise click.ClickException(
            f"C3D cache not found: {cache_path}. "
            "Build with `python -m tools.cache_c3d_stats` first."
        )

    out_root = (
        Path(output_dir) if output_dir
        else env.artifacts_dir / "exp057_capture_domain_probe"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    features, filenames = _load_cache(cache_path)
    labels, class_names, perf_ids = _labels_from_filenames(filenames, target)
    click.echo(f"Loaded {len(features)} samples, {(labels >= 0).sum()} valid labels, "
               f"{len(class_names)} classes: {class_names}")

    result = _leave_one_performer_out(features, labels, perf_ids)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
    y_true = np.asarray(result["y_true"])
    y_pred = np.asarray(result["y_pred"])
    cm = confusion_matrix(y_true, y_pred)
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    click.echo(f"\nOverall LPO accuracy: {result['overall_acc']:.4f}")
    click.echo(f"Overall LPO macro-F1: {f1:.4f}")
    click.echo(f"Confusion matrix (rows=true, cols=pred):")
    click.echo("  " + "  ".join(class_names))
    for i, row in enumerate(cm):
        click.echo(f"  {class_names[i]}: {row.tolist()}")

    click.echo(f"\nPer-performer accuracy spread (first 10):")
    for perf in sorted(result["per_performer_acc"].keys())[:10]:
        click.echo(f"  {perf}: {result['per_performer_acc'][perf]:.3f}")

    # Persist
    payload = {
        "target": target,
        "class_names": class_names,
        "overall_acc": result["overall_acc"],
        "macro_f1": f1,
        "per_performer_acc": result["per_performer_acc"],
        "confusion_matrix": cm.tolist(),
    }
    out_json = out_root / f"probe_{target}.json"
    out_json.write_text(json.dumps(payload, indent=2))
    click.echo(f"\nSaved: {out_json}")

    # Classification report as human-readable md
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0,
    )
    out_md = out_root / f"probe_{target}.md"
    lines = [
        f"# Capture-domain probe — target={target}",
        "",
        f"- Overall LPO accuracy: **{result['overall_acc']:.4f}**",
        f"- Overall LPO macro-F1: **{f1:.4f}**",
        "",
        "## sklearn classification_report",
        "",
        "```",
        report.rstrip(),
        "```",
    ]
    out_md.write_text("\n".join(lines))
    click.echo(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
