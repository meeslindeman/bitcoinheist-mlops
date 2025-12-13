# BitcoinHeist ML Pipeline (Course Project)

This repository implements a production-style ML system for detecting ransomware-related Bitcoin addresses using the BitcoinHeist dataset. The project covers the complete MLOps lifecycle: data ingestion, distributed preprocessing, feature engineering, model training, experiment tracking, API serving, telemetry, monitoring, and testing. The system is designed to be containerized, and runnable end-to-end on a single machine.

---

## Dataset

The pipeline requires the raw BitcoinHeist dataset (CSV format) from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset). Feature definitions are derived from the transaction graph structure around each Bitcoin address.

| Feature | Definition | Usefullness |
| ------- |----------- | ----------- |
| **Income** | Satoshi amount (1 bitcoin = 100 million satoshis) | Captures payment magnitude (ransom payments cluster around specific BTC amounts). |
| **Neighbors** | Number of transactions sending to that address | Ransomware wallets often have few unique payers, unlike exchanges with many. |
| **Weight** | Sum of fractions of starter transactions’ coins reaching the address | Quantifies coin merging behavior (aggregation of payments). |
| **Length** | Longest chain length from a “starter” transaction to the address | Indicates how deep in the transaction graph the address sits (useful for detecting coin-mixing). |
| **Count** | Number of distinct starter transactions connected through chains | Measures how many separate flows converge to that address. |
| **Loop** | Number of starter transactions connected through *multiple* directed paths | Identifies coin-mixing loops. |

## System Architecture Overview

### Offline (Batch) Pipeline
#### 1. Parquet Initialization
- Raw CSV files are converted into Parquet format using `./scripts/csv_to_parquet.py`.

#### 2. Distributed Preprocessing (PySpark)
- Cleans invalid rows
- Enforces schema consistency
- Normalizes raw fields
- Outputs a preprocessed Parquet dataset
  
#### 3. Feature Engineering
- Applies log transforms
- Creates ratios and derived features
- Performs feature selection
- Produces a features Parquet dataset and a JSON feature schema
  
#### 4. Model Training
- Uses Scikit-learn RandomForest Classifier
- Runs cross-validation and evaluation
- Logs metrics and artifacts to MLflow
  
#### 5. Airflow Orchestration
- An Airflow DAG (training_dag.py) coordinates all batch stages
- Each stage is independent and overwrites its own Parquet output

### Online (Serving) Pipeline

#### Inference Endpoint
- Fast prediction endpoint (/`predict`) served via Flask
- Loads the same model artifacts as the training stage
  
#### Monitoring & Metrics
Exposes Prometheus metrics:
- Request count
- Error count
- Latency
= Prediction labels

## Telemetry and Drift Monitoring

### Live Data Simulation

A script samples from the features Parquet to simulate live traffic:
- Generates `telemetry/live_data_dist.json`
- Injects missing values to emulate real-world issues

### Drift Metrics

For selected features, the system computes:
- PSI (Population Stability Index)
- Missing value ratio
- Mean
- Standard deviation

Metrics are pushed to Prometheus via Pushgateway and visualized in Grafana.

## Running the app

### 1. Clone this repository
```bash
git clone https://github.com/meeslindeman/bitcoinheist-mlops.git
cd bitcoinheist-mlops
```

### 2. Download the dataset


Retrieve the dataset using the following command:

```bash
wget -O data/bitcoinheist.zip \
  https://archive.ics.uci.edu/static/public/526/bitcoinheistransomwareaddressdataset.zip && \
unzip data/bitcoinheist.zip -d data && \
rm ./data/bitcoinheist.zip
```

> [!NOTE]
> If you do not have `wget`, you might want to try `curl` instead:
```bash
curl -o data/bitcoinheist.zip \
	https://archive.ics.uci.edu/static/public/526/bitcoinheistransomwareaddressdataset.zip && \
unzip data/bitcoinheist.zip -d data && \
rm ./data/bitcoinheist.zip
```

Make sure the file is now stored at `data/BitcoinHeistData.csv`. This file is consumed by the initialization step (`scripts/csv_to_parquet.py`) to produce the Parquet dataset used throughout preprocessing, feature engineering, and training.

### 3. Build the Docker image for the app
```bash
make build
```

### 4. Run the app and all other containers required for the app
```bash
make up
```

This runs the following Docker containers in the background:
- App → Flask service exposing the /predict inference endpoint
- MLflow → Tracks experiments, metrics, and model artifacts
- Airflow → Orchestrates batch pipeline stages via DAGs
- Prometheus → Collects and stores monitoring metrics
- Alertmanager → Handles alerts triggered by Prometheus rules
- Pushgateway → Accepts ephemeral metrics from batch jobs
- Grafana → Visualizes metrics and dashboards for monitoring

*Steps 1 - 3 are needed once, afterwards you can just start the app using `make up`.*

### 5. Trigger the Airflow training pipeline

Open Airflow UI (http://localhost:4242/dags) and trigger `bitcoin-heist-training`.

Training performs:
- Parquet → preprocessing → feature engineering → model training 
- MLflow logs experiment and model artifacts

Once the training DAG has finished successfully, run:
```bash
make drift
```

This generates `telemetry/live_data_dist.json` by sampling from the features parquet.
A PSI monitor then reads `data_dist.json` and `live_data_dist.json`, computes PSI + missing ratio + mean + std per tracked feature, and pushes metrics to Prometheus via Pushgateway.

### 6. Use the live interface
   
Open http://localhost:5001/ and submit values to receive prediction.

This serves a UI that uses the App's API. To interact with the APi directly, you can use `curl`, e.g.:
```bash
curl \
  -H "Content-Type: application/json" \
  -d '{
        "year": 2014,
        "day": 150,
        "length": 5,
        "weight": 0.12,
        "count": 3,
        "looped": 1,
        "neighbors": 15,
        "income": 0.003
      }' \
  -X POST http://localhost:5001/predict
```

More request can be found in `./scripts/api_requests.sh`.

### 7. Check out Grafana dashboard 

Grafana dashboard is persisted via a bind mount to `infra/grafana/data`, which contains Grafana’s internal SQLite database (`grafana.db`).

Open http://localhost:3000/ and login with default credentials (local development only):
- **Username:** admin  
- **Password:** admin

### 8. (Optional) Track logging
```bash
make logs-app
```

### 9. (Optional) Run tests


#### Unit test

Covers:
- Feature generators
- Preprocessing helpers
- Model training wrapper
- MLflow logging
- Telemetry instrumentation

Target coverage: >80%

Once your container with the app is available, you can run the unittest and view the coverage report:

```bash
make test-unit
```

#### Integration tests

Runs inside Docker Compose and validates:
- API startup
- Model artifact loading
- Prediction contract
- End-to-end inference flow

Command:
```bash
make test-integration
```

### 10. Shutdown system
```bash
make down
```

## References

- Dataset (UCI): https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset  
- Paper (feature definitions and further background): https://arxiv.org/pdf/1906.07852  
- External GitHub implementation (referenced for preprocessing): https://github.com/toji-ut/BitcoinHeistRansomwareAnalytics  
