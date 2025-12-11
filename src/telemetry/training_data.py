import json
import numpy as np
import pandas as pd
from typing import Dict, List

from configs.configs import TelemetryConfig


# note: calculate number of bins using Doane's formula (https://www.nannyml.com/blog/population-stability-index-psi)
def doanes_formula(data: np.array) -> int:
    # note: need at least 3 data points to compute skewness
    if len(data) < 3:
        return 1
    
    skewness = pd.Series(data).skew()

    # note: see https://en.wikipedia.org/wiki/Sturges%27s_rule
    sigma_g1 = np.sqrt((6 * (len(data) - 2)) / ((len(data) + 1) * (len(data) + 3)))
    num_bins = 1 + np.log2(len(data)) + np.log2(1 + abs(skewness) / sigma_g1)

    return max(1, int(round(num_bins)))


def calculate_feature_distribution(data: pd.DataFrame, features: List[str]) -> Dict[str, Dict]:
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

        # note: determine number of bins using Doane's formula
        num_bins = doanes_formula(non_missing.to_numpy())
        counts, bin_edges = np.histogram(non_missing, bins=num_bins)

        dist[feature] = {
            "missing_ratio": missing_ratio,
            "mean": float(non_missing.mean()),
            "std": float(non_missing.std()),
            "hist": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
            "num_bins": num_bins
        }

    return dist


def save_training_distribution(data: pd.DataFrame) -> None:
    features = TelemetryConfig.monitored_features
    dist = calculate_feature_distribution(data, features)

    path = TelemetryConfig.telemetry_training_data_dist_path
    with open(path, "w") as f:
        json.dump(dist, f, indent=2)

