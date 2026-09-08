"""Evaluation metrics for emotion classification.

See: diema_challenge_implementation_spec.md §6.6 — 評価指標
Related: diema/training/evaluate.py, tools/summarize_cv.py
"""

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# Official 12-class emotion labels (index order matches emo_to_idx_12.txt)
EMOTION_LABELS = [
    "anger", "contempt", "disgust", "fear", "joy", "sadness",
    "surprise", "jealousy", "shame", "guilt", "gratitude", "pride",
]

NUM_CLASSES = len(EMOTION_LABELS)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute all standard metrics for the DIEM-A challenge.

    Args:
        y_true: ground truth labels (N,) int array
        y_pred: predicted labels (N,) int array

    Returns:
        Dict with keys:
          - macro_f1: float
          - accuracy: float
          - class_wise_f1: dict[str, float] (per emotion label)
          - confusion_matrix: np.ndarray (NUM_CLASSES x NUM_CLASSES)
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)

    class_wise = {}
    for i, label in enumerate(EMOTION_LABELS):
        class_wise[label] = float(per_class_f1[i]) if i < len(per_class_f1) else 0.0

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "class_wise_f1": class_wise,
        "confusion_matrix": cm,
    }


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute metrics grouped by an attribute (e.g., performer, nationality, intensity).

    Args:
        y_true: ground truth labels (N,)
        y_pred: predicted labels (N,)
        groups: group labels (N,) — e.g., performer IDs or nationality codes

    Returns:
        Dict mapping group_value -> {"macro_f1": float, "accuracy": float, "count": int}
    """
    results = {}
    for group_val in np.unique(groups):
        mask = groups == group_val
        if mask.sum() == 0:
            continue
        gt = y_true[mask]
        pd_arr = y_pred[mask]
        results[str(group_val)] = {
            "macro_f1": float(f1_score(gt, pd_arr, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)),
            "accuracy": float(accuracy_score(gt, pd_arr)),
            "count": int(mask.sum()),
        }
    return results
