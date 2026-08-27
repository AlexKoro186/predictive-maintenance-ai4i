"""Train and save the final predictive-maintenance model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from predictive_maintenance.data import load_data
from predictive_maintenance.evaluate import (
    calculate_metrics,
    find_cost_optimal_threshold,
)
from predictive_maintenance.features import create_features
from predictive_maintenance.train import (
    create_gradient_boosting_model,
    split_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIRECTORY = PROJECT_ROOT / "models"
REPORT_DIRECTORY = PROJECT_ROOT / "reports"

MODEL_PATH = MODEL_DIRECTORY / "failure_predictor.joblib"
METRICS_PATH = REPORT_DIRECTORY / "final_metrics.json"

FALSE_POSITIVE_COST = 100.0
FALSE_NEGATIVE_COST = 5000.0


def train_final_model() -> dict[str, object]:
    """Train, evaluate, and save the selected model."""

    dataframe = load_data()
    features, target = create_features(dataframe)

    (
        training_features_full,
        test_features,
        training_target_full,
        test_target,
    ) = split_data(features, target)

    (
        training_features,
        validation_features,
        training_target,
        validation_target,
    ) = train_test_split(
        training_features_full,
        training_target_full,
        test_size=0.25,
        random_state=42,
        stratify=training_target_full,
    )

    model = create_gradient_boosting_model(training_features)
    model.fit(training_features, training_target)

    validation_probabilities = model.predict_proba(
        validation_features
    )[:, 1]

    threshold, _ = find_cost_optimal_threshold(
        target=validation_target,
        probabilities=validation_probabilities,
        false_positive_cost=FALSE_POSITIVE_COST,
        false_negative_cost=FALSE_NEGATIVE_COST,
    )

    test_probabilities = model.predict_proba(test_features)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)

    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(
            test_target,
            test_predictions,
            labels=[0, 1],
        ).ravel()
    )

    metrics = calculate_metrics(
        model,
        test_features,
        test_target,
        threshold=threshold,
    )

    estimated_cost = (
        false_positive * FALSE_POSITIVE_COST
        + false_negative * FALSE_NEGATIVE_COST
    )

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "model_name": "Gradient boosting",
        "threshold": threshold,
        "feature_columns": features.columns.tolist(),
        "false_positive_cost": FALSE_POSITIVE_COST,
        "false_negative_cost": FALSE_NEGATIVE_COST,
    }

    joblib.dump(artifact, MODEL_PATH)

    report = {
        "model": "Gradient boosting",
        "threshold": float(threshold),
        "metrics": {
            name: float(value)
            for name, value in metrics.items()
        },
        "confusion_matrix": {
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        },
        "cost_assumptions_eur": {
            "false_positive": FALSE_POSITIVE_COST,
            "false_negative": FALSE_NEGATIVE_COST,
        },
        "estimated_test_cost_eur": float(estimated_cost),
        "training_observations": len(training_features),
        "validation_observations": len(validation_features),
        "test_observations": len(test_features),
    }

    METRICS_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


def main() -> None:
    """Run final model training from the command line."""

    report = train_final_model()

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Selected threshold: {report['threshold']:.2f}")
    print(
        "Estimated test cost: "
        f"EUR {report['estimated_test_cost_eur']:,.2f}"
    )


if __name__ == "__main__":
    main()