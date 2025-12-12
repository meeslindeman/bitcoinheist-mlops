# BitcoinHeist ML Pipeline (Course Project)

A compact reference for running and developing the BitcoinHeist end-to-end ML pipeline.  

---

## Data Prep

The system requires the raw BitcoinHeist dataset (CSV format) from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset).

Run the following commands to get the dataset:
```bash
wget -O data/bitcoinheist.zip https://archive.ics.uci.edu/static/public/526/bitcoinheistransomwareaddressdataset.zip

unzip data/bitcoinheist.zip -d data

rm ./data/bitcoinheist.zip
```

Make sure the file is now stored at `data/BitcoinHeistData.csv`. This file is consumed by the initialization step (`csv_to_parquet`) to produce the Parquet dataset used throughout preprocessing, feature engineering, and training.

## Dataset Feature Notes

| Feature | Definition | Usefullness |
| ------- |----------- | ----------- |
| **Income** | Satoshi amount (1 bitcoin = 100 million satoshis) | Captures payment magnitude — ransom payments cluster around specific BTC amounts. |
| **Neighbors** | Number of transactions sending to that address | Ransomware wallets often have few unique payers, unlike exchanges with many. |
| **Weight** | Sum of fractions of starter transactions’ coins reaching the address | Quantifies coin merging behavior (aggregation of payments). |
| **Length** | Longest chain length from a “starter” transaction to the address | Indicates how deep in the transaction graph the address sits (useful for detecting coin-mixing). |
| **Count** | Number of distinct starter transactions connected through chains | Measures how many separate flows converge to that address. |
| **Loop** | Number of starter transactions connected through *multiple* directed paths | Identifies obfuscation or coin-mixing loops. |

## System Architecture Overview

Offline (Batch) Pipeline
1. Parquet initialization
   
    Convert raw CSV → Parquet (`scripts/csv_to_parquet.py`).
2. Distributed preprocessing (PySpark)
   
    Cleans data, enforces schema.
3. Feature engineering
   
    Log transforms, ratios, z-scores, categorical cleanup.
4. Model training
   
    MLflow-tracked experiment.
5. Airflow DAG (`training_dag.py`) orchestrates the full batch pipeline.

Online (HTTP API)
- Fast inference endpoint (`/predict`) served via Flask.
- Loads same model artefacts as training stage.
- Logs inference telemetry to local volume (Prometheus).

## Test Coverage

Unit Tests (`make test`)
Covers:
- Feature generators
- Preprocessing helpers
- Model training wrapper
- MLflow logging
- Telemetry instrumentation

Integration Tests (`make test-integration`)
Runs inside Docker Compose against:
- API container
- Model artefacts
- Example inference requests

Ensures:
- API contract stability
- Correct model loading
- Valid prediction outputs

## Makefile Workflow

Note: everything runs in the background.

1. Run all unit tests
   
Note: make sure `requirements.txt` is also installed in your local machine (via virtual environment).
```bash
make test
```

2. Start full stack (MLflow, Spark, Airflow, API)
```bash
make up
```

3. Trigger the Airflow training pipeline
   
Open Airflow UI (http://localhost:4242/dags) and trigger `bitcoin-heist-training`.

Training performs:
- Parquet → preprocessing → feature engineering → model training 
- MLflow logs experiment and model artifacts

4. Once the training DAG has finished successfully, run:
```bash
make drift
```

This generates `telemetry/live_data_dist.json` by sampling from the features parquet.
A PSI monitor then reads `data_dist.json` and `live_data_dist.json`, computes PSI + missing ratio + mean + std per tracked feature, and pushes metrics to Prometheus via Pushgateway.

5. Run integration tests
```bash
make test-integration
```

6. Use the live API interface
   
Open http://localhost:5001/ and submit values to receive prediction.

7. Check out Grafana dashboard 

Grafana dashboard is persisted via a bind mount to `infra/grafana/data`, which contains Grafana’s internal SQLite database (`grafana.db`).

Open http://localhost:3000/ and login with default credentials (local development only):
- **Username:** admin  
- **Password:** admin

8. Shutdown system
```bash
make down
```

9. (Optional) Track logging
```bash
make logs-app
```

## References

- Dataset (UCI): https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset  
- Paper (feature definitions and further background): https://arxiv.org/pdf/1906.07852  
- External GitHub implementation (referenced for preprocessing): https://github.com/toji-ut/BitcoinHeistRansomwareAnalytics  
