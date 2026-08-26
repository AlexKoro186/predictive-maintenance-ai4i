"""Evaluation utilities for classification models."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate classification metrics at a selected threshold."""

    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

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


def find_cost_optimal_threshold(
    target: pd.Series,
    probabilities: np.ndarray,
    false_positive_cost: float = 100.0,
    false_negative_cost: float = 5000.0,
) -> tuple[float, pd.DataFrame]:
    """Find the threshold with the lowest estimated maintenance cost."""

    rows = []

    for threshold in np.linspace(0.01, 0.99, 99):
        predictions = (probabilities >= threshold).astype(int)

        true_negative, false_positive, false_negative, true_positive = (
            confusion_matrix(
                target,
                predictions,
                labels=[0, 1],
            ).ravel()
        )

        total_cost = (
            false_positive * false_positive_cost
            + false_negative * false_negative_cost
        )

        rows.append(
            {
                "threshold": threshold,
                "true_negative": true_negative,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_positive": true_positive,
                "total_cost": total_cost,
            }
        )

    results = pd.DataFrame(rows)

    best_row = results.loc[results["total_cost"].idxmin()]
    best_threshold = float(best_row["threshold"])

    return best_threshold, results