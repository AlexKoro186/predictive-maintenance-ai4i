"""Functions for loading the AI4I predictive maintenance dataset."""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"


def load_data(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the AI4I dataset locally or download it from UCI."""

    if data_path.exists():
        return pd.read_csv(data_path)

    dataset = fetch_ucirepo(id=601)

    features = dataset.data.features
    targets = dataset.data.targets

    data = pd.concat([features, targets], axis=1)

    data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_path, index=False)

    return data