import json
import logging
import numpy as np

from datetime import datetime, UTC

from configs.configs import PathsConfig, TelemetryConfig
from src.utils.spark_utils import read_data


logger = logging.getLogger(__name__)


# note: generate mocked "live" telemetry data based on preprocessed data
# note: I want to do this in the container instead of a static json file so we need to run a script
def main():  
    preprocessed_spark = read_data(path=PathsConfig.preprocessed_data_path)
    data = preprocessed_spark.toPandas()

    # note: ["income", "neighbors", "weight"]
    features = TelemetryConfig.monitored_features  

    for f in features:
        if f not in data.columns:
            raise ValueError(f"Feature '{f}' not found in preprocessed data.")

    # note: sample rows to act as "live" traffic
    n_live = 1000
    sampled = data.sample(n=n_live, replace=True, random_state=42).reset_index(drop=True)

    # note: inject some missing values to simulate missing features ratio
    missing_prob = 0.05 

    random_generator = np.random.default_rng(seed=42)
    for f in features:
        mask = random_generator.random(n_live) < missing_prob
        sampled.loc[mask, f] = np.nan

    # note: serialize to JSON
    records = []
    now = datetime.now(UTC)
    for i in range(n_live):
        record = {"timestamp": now}
        for f in features:
            value = sampled.loc[i, f]
            record[f] = value
        records.append(record)

    path = TelemetryConfig.telemetry_live_data_dist_path
    with open(path, "w") as f:
        json.dump(records, f, indent=2)

    logger.info(f"[Telemetry] Wrote {len(records)} mocked live records to: {path}")


if __name__ == "__main__":
    main()
