"""Model creation and training utilities."""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from predictive_maintenance.features import build_preprocessor


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create stratified training and test datasets."""

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def create_dummy_model(features: pd.DataFrame) -> Pipeline:
    """Create a baseline model that predicts the majority class."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]
    )


def create_logistic_model(features: pd.DataFrame) -> Pipeline:
    """Create a class-weighted logistic regression model."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )