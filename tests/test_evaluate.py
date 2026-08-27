"""Tests for classification evaluation utilities."""

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from predictive_maintenance.evaluate import (
    calculate_metrics,
    find_cost_optimal_threshold,
)


class ProbabilityModel:
    """Return predefined probabilities for evaluation tests."""

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.asarray(probabilities)

    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> NDArray[np.float64]:
        """Return probabilities in scikit-learn format."""

        probabilities = self.probabilities[: len(features)]

        return np.column_stack(
            [
                1.0 - probabilities,
                probabilities,
            ]
        )


def test_calculate_metrics_uses_selected_threshold() -> None:
    """Changing the threshold should affect classification metrics."""

    model = ProbabilityModel([0.10, 0.60, 0.70, 0.40])
    features = pd.DataFrame({"sensor": [1, 2, 3, 4]})
    target = pd.Series([0, 0, 1, 1])

    metrics = calculate_metrics(
        model,
        features,
        target,
        threshold=0.50,
    )

    assert metrics["accuracy"] == pytest.approx(0.50)
    assert metrics["precision"] == pytest.approx(0.50)
    assert metrics["recall"] == pytest.approx(0.50)
    assert metrics["f1_score"] == pytest.approx(0.50)
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_lower_threshold_increases_recall() -> None:
    """A lower threshold should detect more positive observations."""

    model = ProbabilityModel([0.10, 0.60, 0.70, 0.40])
    features = pd.DataFrame({"sensor": [1, 2, 3, 4]})
    target = pd.Series([0, 0, 1, 1])

    default_metrics = calculate_metrics(
        model,
        features,
        target,
        threshold=0.50,
    )
    lower_threshold_metrics = calculate_metrics(
        model,
        features,
        target,
        threshold=0.30,
    )

    assert (
        lower_threshold_metrics["recall"]
        > default_metrics["recall"]
    )


def test_find_cost_optimal_threshold_minimizes_cost() -> None:
    """Threshold selection should identify a zero-cost separation."""

    target = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.10, 0.40, 0.60, 0.90])

    threshold, results = find_cost_optimal_threshold(
        target=target,
        probabilities=probabilities,
        false_positive_cost=100.0,
        false_negative_cost=5000.0,
    )

    assert 0.40 < threshold <= 0.60
    assert results["total_cost"].min() == pytest.approx(0.0)