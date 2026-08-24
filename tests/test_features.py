"""Tests for feature creation."""

import pandas as pd

from predictive_maintenance.features import create_features


def test_create_features_removes_leakage_columns() -> None:
    data = pd.DataFrame(
        {
            "Type": ["L", "H"],
            "Torque [Nm]": [40.0, 60.0],
            "Machine failure": [0, 1],
            "TWF": [0, 0],
            "HDF": [0, 1],
            "PWF": [0, 0],
            "OSF": [0, 0],
            "RNF": [0, 0],
        }
    )

    features, target = create_features(data)

    assert "Machine failure" not in features.columns
    assert "TWF" not in features.columns
    assert "HDF" not in features.columns
    assert "PWF" not in features.columns
    assert "OSF" not in features.columns
    assert "RNF" not in features.columns

    assert features.columns.tolist() == ["Type", "Torque [Nm]"]
    assert target.tolist() == [0, 1]