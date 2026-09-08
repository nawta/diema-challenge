"""Test inference pipeline.

Supports single-checkpoint and fold-ensemble inference.

See: diema_challenge_implementation_spec.md §6.7 — 推論
Related: diema/training/train.py, tools/infer_test.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from diema.data.parser import IDX_TO_EMOTION

logger = logging.getLogger(__name__)


def infer_single_checkpoint(
    trainer: pl.Trainer,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
) -> pd.DataFrame:
    """Run inference with a single checkpoint.

    Args:
        trainer: Lightning Trainer
        model: trained LightningModel
        datamodule: DataModule with test data

    Returns:
        DataFrame with predictions.
    """
    predictions = trainer.predict(model, datamodule=datamodule)
    return _predictions_to_dataframe(predictions)


def infer_fold_ensemble(
    fold_predictions: list[pd.DataFrame],
    ensemble_method: str = "mean_prob",
) -> pd.DataFrame:
    """Ensemble predictions from multiple folds.

    Args:
        fold_predictions: list of DataFrames from infer_single_checkpoint
        ensemble_method: 'mean_prob' (average probabilities) or 'majority_vote'

    Returns:
        Ensembled prediction DataFrame.
    """
    if not fold_predictions:
        raise ValueError("No fold predictions to ensemble")

    if ensemble_method == "mean_prob":
        return _ensemble_mean_prob(fold_predictions)
    elif ensemble_method == "majority_vote":
        return _ensemble_majority_vote(fold_predictions)
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")


def _predictions_to_dataframe(predictions: list[tuple]) -> pd.DataFrame:
    """Convert Lightning predict_step output to DataFrame."""
    all_predicted = []
    all_names = []
    all_probas = []

    for predicted, proba, _labels, sample_names in predictions:
        all_predicted.extend(predicted.cpu().numpy().tolist())
        all_names.extend(sample_names)
        all_probas.extend(proba.cpu().numpy().tolist())

    df = pd.DataFrame({
        "sample_name": all_names,
        "predicted_label": all_predicted,
    })

    df["predicted_emotion"] = df["predicted_label"].map(IDX_TO_EMOTION)

    for i, emotion in IDX_TO_EMOTION.items():
        df[f"prob_{emotion}"] = [p[i] for p in all_probas]

    return df


def _ensemble_mean_prob(fold_predictions: list[pd.DataFrame]) -> pd.DataFrame:
    """Ensemble by averaging predicted probabilities."""
    prob_cols = [c for c in fold_predictions[0].columns if c.startswith("prob_")]
    all_sample_names = fold_predictions[0]["sample_name"].values

    # Average probabilities across folds
    prob_sum = np.zeros((len(all_sample_names), len(prob_cols)))
    for fold_df in fold_predictions:
        fold_probs = fold_df[prob_cols].values
        prob_sum += fold_probs
    avg_probs = prob_sum / len(fold_predictions)

    # Argmax for final prediction
    predicted_labels = avg_probs.argmax(axis=1)

    result = pd.DataFrame({
        "sample_name": all_sample_names,
        "predicted_label": predicted_labels,
        "predicted_emotion": [IDX_TO_EMOTION[l] for l in predicted_labels],
    })

    for i, col in enumerate(prob_cols):
        result[col] = avg_probs[:, i]

    return result


def _ensemble_majority_vote(fold_predictions: list[pd.DataFrame]) -> pd.DataFrame:
    """Ensemble by majority voting."""
    from collections import Counter

    all_sample_names = fold_predictions[0]["sample_name"].values
    votes = {name: [] for name in all_sample_names}

    for fold_df in fold_predictions:
        for _, row in fold_df.iterrows():
            votes[row["sample_name"]].append(row["predicted_label"])

    predicted_labels = []
    for name in all_sample_names:
        counter = Counter(votes[name])
        predicted_labels.append(counter.most_common(1)[0][0])

    return pd.DataFrame({
        "sample_name": all_sample_names,
        "predicted_label": predicted_labels,
        "predicted_emotion": [IDX_TO_EMOTION[l] for l in predicted_labels],
    })
