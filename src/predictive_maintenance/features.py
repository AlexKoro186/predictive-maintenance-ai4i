"""Feature selection and preprocessing."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
    """Separate model features and target without data leakage."""

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")

    columns_to_remove = [
        TARGET_COLUMN,
        *IDENTIFIER_COLUMNS,
        *LEAKAGE_COLUMNS,
    ]

    features = data.drop(columns=columns_to_remove, errors="ignore")
    target = data[TARGET_COLUMN].astype(int).copy()

    return features, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing steps for numerical and categorical features."""

    categorical_columns = features.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_columns = features.select_dtypes(
        include=["number"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                numerical_columns,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
        ]
    )

    return preprocessor