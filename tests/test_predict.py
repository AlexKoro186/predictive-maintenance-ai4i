"""Tests for machine-failure predictions."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from predictive_maintenance.predict import (
    predict_failures,
    predict_single_machine,
)


class StubModel:
    """Return deterministic probabilities for prediction tests."""

    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """Return one probability pair per observation."""

        failure_probabilities = np.array([0.05, 0.20])[
            : len(features)
        ]

        return np.column_stack(
            [
                1.0 - failure_probabilities,
                failure_probabilities,
            ]
        )


@pytest.fixture
def artifact() -> dict[str, Any]:
    """Create a small model artifact for unit tests."""

    return {
        "model": StubModel(),
        "threshold": 0.07,
        "feature_columns": ["sensor_a", "sensor_b"],
    }


def test_predict_failures_applies_threshold(
    artifact: dict[str, Any],
) -> None:
    """Predictions should use the stored decision threshold."""

    machine_data = pd.DataFrame(
        {
            "sensor_a": [1.0, 2.0],
            "sensor_b": [3.0, 4.0],
        }
    )

    predictions = predict_failures(
        machine_data,
        artifact=artifact,
    )

    assert predictions["failure_probability"].tolist() == [
        0.05,
        0.20,
    ]
    assert predictions["maintenance_alert"].tolist() == [
        False,
        True,
    ]


def test_predict_single_machine_returns_result(
    artifact: dict[str, Any],
) -> None:
    """A single observation should return a prediction dictionary."""

    result = predict_single_machine(
        {
            "sensor_a": 1.0,
            "sensor_b": 3.0,
        },
        artifact=artifact,
    )

    assert result["failure_probability"] == pytest.approx(0.05)
    assert result["decision_threshold"] == pytest.approx(0.07)
    assert result["maintenance_alert"] is False


def test_predict_failures_rejects_missing_columns(
    artifact: dict[str, Any],
) -> None:
    """Missing machine features should produce a clear error."""

    machine_data = pd.DataFrame(
        {
            "sensor_a": [1.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing feature columns: sensor_b",
    ):
        predict_failures(
            machine_data,
            artifact=artifact,
        )