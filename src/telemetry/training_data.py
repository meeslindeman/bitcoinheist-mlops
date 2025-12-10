import json
import numpy as np
import pandas as pd
from typing import Dict, List

from configs.configs import TelemetryConfig


def calculate_feature_distribution(data: pd.DataFrame, features: List[str], num_bins: int = 10) -> Dict[str, Dict[str, List[float]]]:
    dist = {}

    for feature in features:
        series = data[feature]

        missing_ratio = float(series.isna().mean())
        non_missing = series.dropna()

        if non_missing.empty:
            # note: fall back, no real data
            dist[feature] = {
                "missing_ratio": missing_ratio,
                "mean": None,
                "std": None,
                "hist": None,
                "bin_edges": None,
            }
            continue

        counts, bin_edges = np.histogram(non_missing, bins=num_bins)

        dist[feature] = {
            "missing_ratio": missing_ratio,
            "mean": float(non_missing.mean()),
            "std": float(non_missing.std()),
            "hist": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
        }

    return dist


def save_training_distribution(data: pd.DataFrame) -> None:
    features = TelemetryConfig.monitored_features
    dist = calculate_feature_distribution(data, features)

    path = TelemetryConfig.telemetry_training_data_dist_path
    with open(path, "w") as f:
        json.dump(dist, f, indent=2)

