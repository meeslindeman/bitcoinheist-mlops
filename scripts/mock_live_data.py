import json
from datetime import datetime

import numpy as np
import pandas as pd

from configs.configs import PathsConfig, TelemetryConfig
from src.spark_utils import read_data


def main():  
    features_spark = read_data(path=PathsConfig.features_data_path)
    df = features_spark.toPandas()

    features = TelemetryConfig.monitored_features
    n_live = 1000
    sampled = df.sample(n=n_live, replace=True, random_state=42).reset_index(drop=True)

    records = []
    now = datetime.utcnow()
    for i in range(n_live):
        rec = {"timestamp": now.isoformat()}
        for f in features:
            val = sampled.loc[i, f]
            if isinstance(val, (np.generic,)):
                val = val.item()
            rec[f] = val
        records.append(rec)

    path = TelemetryConfig.telemetry_live_data_dist_path
    with open(path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"[mock-live] Wrote {len(records)} mocked live records to: {path}")


if __name__ == "__main__":
    main()
