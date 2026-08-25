"""Evaluation utilities for classification models."""

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    """Calculate classification metrics."""

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    return {
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(
            target,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            target,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            target,
            predictions,
            zero_division=0,
        ),
        "pr_auc": average_precision_score(target, probabilities),
    }