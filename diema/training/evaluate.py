"""Evaluation utilities for OOF predictions and analysis.

Computes metrics at multiple granularities: overall, per-class,
per-performer, per-nationality, per-intensity.

See: diema_challenge_implementation_spec.md §6.6 — 評価
Related: utils/metrics.py, tools/summarize_cv.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from diema.data.parser import parse_filename, IDX_TO_EMOTION
from utils.metrics import compute_metrics, compute_group_metrics

logger = logging.getLogger(__name__)


def evaluate_oof(oof_df: pd.DataFrame) -> dict:
    """Evaluate OOF predictions with comprehensive metrics.

    Args:
        oof_df: DataFrame with columns:
          - sample_name: filename stem
          - true_label: ground truth label index
          - predicted_label: predicted label index

    Returns:
        Dict with overall + grouped metrics.
    """
    y_true = oof_df["true_label"].values
    y_pred = oof_df["predicted_label"].values

    overall = compute_metrics(y_true, y_pred)

    # Parse metadata from filenames for grouped analysis
    performers = []
    nationalities = []
    intensities = []
    for name in oof_df["sample_name"]:
        try:
            info = parse_filename(name)
            performers.append(info.performer_id)
            nationalities.append(info.nationality)
            intensities.append(info.intensity)
        except (ValueError, IndexError):
            performers.append("unknown")
            nationalities.append("unknown")
            intensities.append("unknown")

    performer_metrics = compute_group_metrics(y_true, y_pred, np.array(performers))
    nationality_metrics = compute_group_metrics(y_true, y_pred, np.array(nationalities))
    intensity_metrics = compute_group_metrics(y_true, y_pred, np.array(intensities))

    return {
        "overall": overall,
        "per_performer": performer_metrics,
        "per_nationality": nationality_metrics,
        "per_intensity": intensity_metrics,
    }


def format_evaluation_report(eval_results: dict) -> str:
    """Format evaluation results as a human-readable report.

    Args:
        eval_results: output from evaluate_oof()

    Returns:
        Formatted string report.
    """
    lines = []
    overall = eval_results["overall"]
    lines.append("=== Overall Metrics ===")
    lines.append(f"  Macro-F1:  {overall['macro_f1']:.4f}")
    lines.append(f"  Accuracy:  {overall['accuracy']:.4f}")

    lines.append("\n=== Per-Class F1 ===")
    for emotion, f1 in sorted(overall["class_wise_f1"].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {emotion:12s}: {f1:.4f}")

    for group_name, group_metrics in [
        ("Nationality", eval_results.get("per_nationality", {})),
        ("Intensity", eval_results.get("per_intensity", {})),
    ]:
        if group_metrics:
            lines.append(f"\n=== Per-{group_name} ===")
            for gval, gm in sorted(group_metrics.items()):
                lines.append(f"  {gval:12s}: F1={gm['macro_f1']:.4f}  Acc={gm['accuracy']:.4f}  n={gm['count']}")

    return "\n".join(lines)


def save_oof_predictions(
    predictions: list[tuple],
    output_path: str | Path,
) -> pd.DataFrame:
    """Save OOF predictions from Lightning predict_step output.

    Args:
        predictions: list of (predicted, proba, labels, sample_names) tuples
            from LightningModel.predict_step
        output_path: CSV output path

    Returns:
        DataFrame with OOF predictions.
    """
    all_predicted = []
    all_labels = []
    all_names = []
    all_probas = []

    for predicted, proba, labels, sample_names in predictions:
        all_predicted.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_names.extend(sample_names)
        all_probas.extend(proba.cpu().numpy().tolist())

    df = pd.DataFrame({
        "sample_name": all_names,
        "true_label": all_labels,
        "predicted_label": all_predicted,
    })

    # Add emotion name columns
    df["true_emotion"] = df["true_label"].map(IDX_TO_EMOTION)
    df["predicted_emotion"] = df["predicted_label"].map(IDX_TO_EMOTION)

    # Add probability columns
    for i, emotion in IDX_TO_EMOTION.items():
        df[f"prob_{emotion}"] = [p[i] for p in all_probas]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} OOF predictions to {output_path}")

    return df
