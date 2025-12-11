import json
import logging
import math
from typing import Dict

import numpy as np
import pandas as pd
from prometheus_client import Gauge, CollectorRegistry, push_to_gateway

from configs.configs import TelemetryConfig


logger = logging.getLogger(__name__)


def get_psi(training_percentages: np.ndarray, latest_percentages: np.ndarray) -> float:
    psi = 0.0
    for training, latest in zip(training_percentages, latest_percentages):
        psi += (latest - training) * math.log(latest / training)
    return psi


def load_training_dist() -> Dict:
    with open(TelemetryConfig.telemetry_training_data_dist_path, "r") as f:
        return json.load(f)


def load_live_data() -> pd.DataFrame:
    with open(TelemetryConfig.telemetry_live_data_dist_path, "r") as f:
        records = json.load(f)
    if not records:
        raise RuntimeError("No live telemetry records found in live_data_dist.json.")
    data = pd.DataFrame.from_records(records)
    # note: not strictly necessary as we mock data ourselves
    data.sort_values("timestamp", ascending=False, inplace=True)
    return data


def main():
    training_dist = load_training_dist()
    live_data = load_live_data()

    registry = CollectorRegistry()
    psi_gauge = Gauge(
        "feature_psi",
        "PSI between training and live data",
        labelnames=["feature"],
        registry=registry,
    )
    missing_gauge = Gauge(
        "feature_missing_ratio",
        "Fraction of missing values per feature",
        labelnames=["feature"],
        registry=registry,
    )

    for feature in TelemetryConfig.monitored_features:
        # note: if feature not found, we skip it
        if feature not in training_dist or feature not in live_data.columns:
            logger.warning(f"Feature {feature} not found in training distribution or live data.")
            continue

        missing_ratio = float(live_data[feature].isna().mean())
        missing_gauge.labels(feature=feature).set(missing_ratio)

        # note: calculate PSI
        training_info = training_dist[feature]
        hist = np.array(training_info["hist"])
        # note: use same bin edges as training data
        bin_edges = np.array(training_info["bin_edges"])

        live_values = live_data[feature].dropna()
        # note: check for empty live values
        if live_values.empty:
            logger.warning(f"No live data available for feature {feature}. Skipping PSI calculation.")
            continue

        live_counts, _ = np.histogram(live_values, bins=bin_edges)
        # note: skip if no live data falls into the bins
        if live_counts.sum() == 0:
            logger.warning(f"No live data falls into the bins for feature {feature}. Skipping PSI calculation.")
            continue
        
        epislon = 1e-6

        training_percentages = hist / hist.sum() + epislon
        live_percentages = live_counts / live_counts.sum() + epislon

        psi = get_psi(training_percentages, live_percentages)
        psi_gauge.labels(feature=feature).set(float(psi))

    push_to_gateway(
        TelemetryConfig.push_gateway_url,
        job="drift_monitoring",
        registry=registry,
    )

    logger.info("[Telemetry] Pushed telemetry metrics to Pushgateway.")

    
if __name__ == "__main__":
    main()