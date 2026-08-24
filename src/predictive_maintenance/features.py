"""Feature selection for machine-failure prediction."""

import pandas as pd


TARGET_COLUMN = "Machine failure"

IDENTIFIER_COLUMNS = [
    "UDI",
    "Product ID",
]

LEAKAGE_COLUMNS = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def create_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target while preventing data leakage."""

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")

    columns_to_drop = [
        TARGET_COLUMN,
        *IDENTIFIER_COLUMNS,
        *LEAKAGE_COLUMNS,
    ]

    features = data.drop(columns=columns_to_drop, errors="ignore")
    target = data[TARGET_COLUMN].copy()

    return features, target