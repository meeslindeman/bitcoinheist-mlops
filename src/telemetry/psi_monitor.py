import json
import logging
import math
from time import time
from typing import Dict, List, Optional

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
    path = TelemetryConfig.telemetry_training_data_dist_path
    with open(path, "r") as f:
        return json.load(f)


def load_live_df() -> pd.DataFrame:
    path = TelemetryConfig.telemetry_live_data_dist_path
    with open(path, "r") as f:
        records = json.load(f)
    if not records:
        raise RuntimeError("No live telemetry records found in live_data_dist.json.")
    df = pd.DataFrame.from_records(records)
    df.sort_values("timestamp", ascending=False, inplace=True)
    return df


def compute_live_stats(df: pd.DataFrame, features: List[str]) -> Dict:
    stats: Dict[str, Dict] = {}
    for feature in features:
        if feature not in df.columns:
            logger.warning(f"Feature '{feature}' not in live telemetry; skipping stats.")
            continue
        series = df[feature]
        missing_ratio = float(series.isna().mean())
        non_missing = series.dropna()

        if non_missing.empty:
            stats[feature] = {
                "missing_ratio": missing_ratio,
                "mean": None,
                "std": None,
            }
        else:
            stats[feature] = {
                "missing_ratio": missing_ratio,
                "mean": float(non_missing.mean()),
                "std": float(non_missing.std()),
            }
    return stats


def compute_psi_for_feature(
    feature: str,
    training_info: Dict,
    live_series: pd.Series,
) -> Optional[float]:
    hist = training_info["hist"]
    bin_edges = training_info["bin_edges"]

    if hist is None or bin_edges is None:
        logger.warning(f"No histogram for feature '{feature}' in training dist; PSI skipped.")
        return None

    hist = np.array(hist, dtype=float)
    training_total = hist.sum()
    if training_total == 0:
        logger.warning(f"Training histogram for feature '{feature}' has zero total count; PSI skipped.")
        return None

    # Training percentages + epsilon
    training_percentages = hist / training_total
    training_percentages = training_percentages + TelemetryConfig.epsilon

    # Live distribution using same bins
    non_missing_live = live_series.dropna()
    if non_missing_live.empty:
        logger.warning(f"No non-missing live values for feature '{feature}'; PSI skipped.")
        return None

    live_counts, _ = np.histogram(non_missing_live, bins=np.array(bin_edges))
    live_total = live_counts.sum()
    if live_total == 0:
        logger.warning(f"Live histogram for feature '{feature}' has zero total count; PSI skipped.")
        return None

    latest_percentages = live_counts / live_total
    latest_percentages = latest_percentages + TelemetryConfig.epsilon

    return float(get_psi(training_percentages, latest_percentages))


def run_psi_monitor(): 
    features = TelemetryConfig.monitored_features

    training_dist = load_training_dist()
    live_df = load_live_df()

    n = TelemetryConfig.num_instances_for_live_dist
    live_window = live_df.head(n)
    if live_window.shape[0] < n:
        logger.warning(
            f"Only {live_window.shape[0]} live rows available, less than required {n}."
        )

    live_stats = compute_live_stats(live_window, features)

    psi_values: Dict[str, Optional[float]] = {}
    for feature in features:
        if feature not in training_dist:
            logger.warning(f"Feature '{feature}' missing from training dist; skipping PSI.")
            continue
        if feature not in live_window.columns:
            logger.warning(f"Feature '{feature}' missing from live data; skipping PSI.")
            continue

        psi = compute_psi_for_feature(feature, training_dist[feature], live_window[feature])
        psi_values[feature] = psi

    registry = CollectorRegistry()

    psi_gauge = Gauge(
        "feature_psi",
        "PSI between training and live data for numeric features",
        labelnames=["feature"],
        registry=registry,
    )
    miss_gauge = Gauge(
        "feature_missing_ratio",
        "Fraction of missing values per feature over monitoring window",
        labelnames=["feature"],
        registry=registry,
    )

    for feature in features:
        # PSI
        psi = psi_values.get(feature)
        if psi is not None:
            psi_gauge.labels(feature=feature).set(psi)

        # missing / mean / std
        stats = live_stats.get(feature)
        if stats is None:
            continue

        miss_gauge.labels(feature=feature).set(float(stats["missing_ratio"]))

    drift_job_last_success = Gauge(
        "drift_job_last_success_timestamp_seconds",
        "Unix timestamp of the last successful drift monitoring job run",
        registry=registry,
    )
    drift_job_last_success.set(time())

    push_to_gateway(
        TelemetryConfig.push_gateway_url,
        job="drift_monitoring",
        registry=registry,
    )

    print("[psi-monitor] Pushed PSI + live stats to Pushgateway.")

def main():
    run_psi_monitor()
    
if __name__ == "__main__":
    main()