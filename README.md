# BitcoinHeist ML Pipeline (Course Project)

This repository implements a fully local, production-style ML system for detecting ransomware-related Bitcoin addresses using the BitcoinHeist dataset.  
The project covers the complete MLOps lifecycle: data ingestion, distributed preprocessing, feature engineering, model training, experiment tracking, API serving, telemetry, monitoring, and testing.
The system is designed to be containerized, and runnable end-to-end on a single machine without external services.

---

## Dataset

The system requires the raw BitcoinHeist dataset (CSV format) from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset).

Run the following commands to get the dataset:
```bash
wget -O data/bitcoinheist.zip https://archive.ics.uci.edu/static/public/526/bitcoinheistransomwareaddressdataset.zip

unzip data/bitcoinheist.zip -d data

rm ./data/bitcoinheist.zip
```

Make sure the file is now stored at `data/BitcoinHeistData.csv`. This file is consumed by the initialization step (`csv_to_parquet`) to produce the Parquet dataset used throughout preprocessing, feature engineering, and training.

## Dataset Feature Notes

Feature definitions are derived from the transaction graph structure around each Bitcoin address.

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
    - Convert raw CSV → Parquet (`scripts/csv_to_parquet.py`).
2. Distributed preprocessing (PySpark)
    - Cleans invalid rows
    - Enforces schema
    - Normalizes raw fields
    - Outputs preprocessed Parquet dataset
3. Feature engineering
    - Log transforms
    - Ratios and derived features
    - Feature selection
    - Outputs a features Parquet dataset and a JSON feature schema
4. Model training
    - Scikit-learn `RandomForrest` Classifier
    - Cross validation and evaluation
    - Metrics and artifacts logged to MLflow
5. Airflow Orchestration
    - Ariflow DAG (`training_dag.py`) coordinates all batch stages
    - Each stage is separate and overwrites its own Parquet output

Online (Serving) Pipeline
- Fast inference endpoint (`/predict`) served via Flask.
- Loads same model artefacts as training stage.
- Loads:
    - Trained model
    - Feature schema
    - Preprocessing logic shared with training
- Exposes Prometheus metrics for:
    - Request count
    - Error count
    - Latency
    - Prediction labels

## Telemetry and Drift Monitoring

**Live Data Simulation**

A script samples from the features Parquet to simulate live traffic:
- Generates `telemetry/live_data_dist.json`
- Injects missing values to emulate real-world issues

**Drift Metrics**

For selected features, the system computes:
- PSI (Population Stability Index)
- Missing value ratio
- Mean
- Standard deviation

Metrics are pushed to Prometheus via Pushgateway and visualized in Grafana.

## Test Coverage

**Unit Tests** 

Command:
```bash
make test
```
Covers:
- Feature generators
- Preprocessing helpers
- Model training wrapper
- MLflow logging
- Telemetry instrumentation

Target coverage: >80%

**Integration Tests** 

Command:
```bash
make test-integration
```

Runs inside Docker Compose and validates:
- API startup
- Model artifact loading
- Prediction contract
- End-to-end inference flow

## Local Development Workflow

All services run in the background.


1. Install Python dependencies locally (virtualenv recommended) via `requirements.txt`.

2. Run unit tests
```bash
make test
```

3. Start full stack
```bash
make up
```

This builds the Docker image and starts:
- Spark
- MLflow
- Airflow
- Flask API
- Prometheus
- Pushgateway
- Grafana

4. Trigger the Airflow training pipeline
   
Open Airflow UI (http://localhost:4242/dags) and trigger `bitcoin-heist-training`.

Training performs:
- Parquet → preprocessing → feature engineering → model training 
- MLflow logs experiment and model artifacts

5. Once the training DAG has finished successfully, run:
```bash
make drift
```

This generates `telemetry/live_data_dist.json` by sampling from the features parquet.
A PSI monitor then reads `data_dist.json` and `live_data_dist.json`, computes PSI + missing ratio + mean + std per tracked feature, and pushes metrics to Prometheus via Pushgateway.

6. Run integration tests
```bash
make test-integration
```

7. Use the live API interface
   
Open http://localhost:5001/ and submit values to receive prediction.

8. Check out Grafana dashboard 

Grafana dashboard is persisted via a bind mount to `infra/grafana/data`, which contains Grafana’s internal SQLite database (`grafana.db`).

Open http://localhost:3000/ and login with default credentials (local development only):
- **Username:** admin  
- **Password:** admin

9. Shutdown system
```bash
make down
```

10. (Optional) Track logging
```bash
make logs-app
```

## Design Notes

- PySpark is used only where distributed processing provides value; model training runs locally.
- Parquet is used throughout for efficient columnar storage and schema enforcement.
- Training and serving share the same feature and preprocessing code to avoid training-serving skew.
- All components are containerized and orchestrated via Docker Compose.

## References

- Dataset (UCI): https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset  
- Paper (feature definitions and further background): https://arxiv.org/pdf/1906.07852  
- External GitHub implementation (referenced for preprocessing): https://github.com/toji-ut/BitcoinHeistRansomwareAnalytics  
