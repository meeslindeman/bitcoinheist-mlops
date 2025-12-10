from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass
class PathsConfig:
    raw_csv_path: str = str(PROJECT_ROOT / "data" / "BitcoinHeistData.csv")

    raw_data_path: str = str(PROJECT_ROOT / "data" / "parquet" / "raw")
    preprocessed_data_path: str = str(PROJECT_ROOT / "data" / "parquet" / "preprocessed")
    features_data_path: str = str(PROJECT_ROOT / "data" / "parquet" / "features")

    model_path: str = str(PROJECT_ROOT / "models")  
    model_file_path: str = str(Path(model_path) / "BitcoinHeist.pkl") 
    feature_columns_path: str = str(Path(model_path) / "feature_columns.json")


class TelemetryConfig:
    # note: presumably most important features to monitor for data drift
    monitored_features = ["income", "neighbors", "weight"]  
    epsilon = 1e-6
    num_instances_for_live_dist = 10

    # note: paths to store data distributions
    telemetry_training_data_dist_path = str(PROJECT_ROOT / "telemetry" / "data_dist.json")
    telemetry_live_data_dist_path: str = str(PROJECT_ROOT / "telemetry" / "live_data_dist.json")

    push_gateway_url: str = "http://pushgateway:9091"


@dataclass(init=False, frozen=True)
class RunConfig:
    random_seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    num_folds: int = 3
    evaluate_model: bool = True

    log_to_mlflow: bool = True 
    mlflow_tracking_uri: str = "http://mlflow:8080"
    experiment_name: str = "BitcoinHeist"
    run_name: str = "training_run"

    sample_rate: float = 0.2


@dataclass(init=False, frozen=True)
class ModelConfig:
    model_name: str = "BitcoinHeist"
    n_estimators: int = 20
    max_depth: int = 20
    random_state: int = 42