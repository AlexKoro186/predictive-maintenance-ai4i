"""Generate maintenance predictions for new machine observations."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "failure_predictor.joblib"


def load_model_artifact(
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Load the trained model and its decision configuration."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            "Run the final training script first."
        )

    artifact: dict[str, Any] = joblib.load(model_path)
    return artifact


def predict_failures(
    machine_data: pd.DataFrame,
    artifact: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Predict failure probabilities and maintenance alerts."""

    active_artifact = artifact or load_model_artifact()

    model = active_artifact["model"]
    threshold = float(active_artifact["threshold"])
    feature_columns = list(active_artifact["feature_columns"])

    missing_columns = [
        column
        for column in feature_columns
        if column not in machine_data.columns
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing feature columns: {missing}")

    ordered_data = machine_data.loc[:, feature_columns]
    probabilities = model.predict_proba(ordered_data)[:, 1]
    alerts = probabilities >= threshold

    return pd.DataFrame(
        {
            "failure_probability": probabilities,
            "decision_threshold": threshold,
            "maintenance_alert": alerts,
        },
        index=machine_data.index,
    )


def predict_single_machine(
    machine_data: Mapping[str, object],
    artifact: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Predict the failure risk for one machine observation."""

    prediction = predict_failures(
        pd.DataFrame([machine_data]),
        artifact=artifact,
    ).iloc[0]

    return {
        "failure_probability": float(
            prediction["failure_probability"]
        ),
        "decision_threshold": float(
            prediction["decision_threshold"]
        ),
        "maintenance_alert": bool(
            prediction["maintenance_alert"]
        ),
    }


def parse_arguments() -> argparse.Namespace:
    """Parse machine measurements from the command line."""

    parser = argparse.ArgumentParser(
        description="Predict whether a machine requires maintenance."
    )

    parser.add_argument(
        "--type",
        dest="machine_type",
        choices=["L", "M", "H"],
        required=True,
        help="Product quality type.",
    )
    parser.add_argument(
        "--air-temperature",
        type=float,
        required=True,
        help="Air temperature in Kelvin.",
    )
    parser.add_argument(
        "--process-temperature",
        type=float,
        required=True,
        help="Process temperature in Kelvin.",
    )
    parser.add_argument(
        "--rotational-speed",
        type=float,
        required=True,
        help="Rotational speed in revolutions per minute.",
    )
    parser.add_argument(
        "--torque",
        type=float,
        required=True,
        help="Torque in newton metres.",
    )
    parser.add_argument(
        "--tool-wear",
        type=float,
        required=True,
        help="Tool wear in minutes.",
    )

    return parser.parse_args()


def main() -> None:
    """Run a prediction from command-line measurements."""

    arguments = parse_arguments()

    machine_data = {
    "Type": arguments.machine_type,
    "Air temperature": arguments.air_temperature,
    "Process temperature": arguments.process_temperature,
    "Rotational speed": arguments.rotational_speed,
    "Torque": arguments.torque,
    "Tool wear": arguments.tool_wear,
}

    prediction = predict_single_machine(machine_data)

    probability = float(prediction["failure_probability"])
    threshold = float(prediction["decision_threshold"])
    alert = bool(prediction["maintenance_alert"])

    print(f"Failure probability: {probability:.2%}")
    print(f"Decision threshold: {threshold:.2f}")
    print(f"Maintenance alert: {'Yes' if alert else 'No'}")


if __name__ == "__main__":
    main()