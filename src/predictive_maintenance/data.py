"""Load the AI4I 2020 predictive maintenance dataset."""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

PROJECT_ROOT = Path.cwd().resolve()
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"


def load_data(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the dataset from disk or download it from UCI."""

    if data_path.exists():
        print(f"Loading local dataset from {data_path}")
        return pd.read_csv(data_path)

    print("Downloading AI4I dataset from UCI...")

    dataset = fetch_ucirepo(id=601)

    features = dataset.data.features
    targets = dataset.data.targets

    data = pd.concat([features, targets], axis=1)

    data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_path, index=False)

    print(f"Dataset saved to {data_path}")

    return data