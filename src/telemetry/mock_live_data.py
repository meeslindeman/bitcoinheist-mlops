import json
from datetime import datetime

import numpy as np

from configs.configs import PathsConfig, TelemetryConfig
from src.utils.spark_utils import read_data


# note: generate mocked "live" telemetry data based on preprocessed data
# note: I want to do this in the container instead of a static json file so we need to run a script
def main():  
    preprocessed_spark = read_data(path=PathsConfig.preprocessed_data_path)
    df = preprocessed_spark.toPandas()

    # note: ["income", "neighbors", "weight"]
    features = TelemetryConfig.monitored_features  

    for f in features:
        if f not in df.columns:
            raise ValueError(f"Feature '{f}' not found in preprocessed data.")

    # note: sample rows to act as "live" traffic
    n_live = 1000
    sampled = df.sample(n=n_live, replace=True, random_state=42).reset_index(drop=True)

    # note: inject some missing values to simulate missing features ratio
    missing_prob = 0.05 

    rng = np.random.default_rng(seed=123)
    for f in features:
        mask = rng.random(n_live) < missing_prob
        # Ensure not all values become missing for that feature (PSI needs some non-missing)
        if mask.all():
            mask[rng.integers(0, n_live)] = False
        sampled.loc[mask, f] = np.nan

    # note: serialize to JSON
    records = []
    now = datetime.utcnow().isoformat()
    for i in range(n_live):
        rec = {"timestamp": now}
        for f in features:
            val = sampled.loc[i, f]
            rec[f] = val
        records.append(rec)

    path = TelemetryConfig.telemetry_live_data_dist_path
    with open(path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"[mock-live] Wrote {len(records)} mocked live records to: {path}")


if __name__ == "__main__":
    main()
