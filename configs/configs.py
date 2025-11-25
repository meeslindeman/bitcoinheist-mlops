from dataclasses import dataclass
import os

@dataclass(init=False, frozen=True)
class PathsConfig:
    raw_data_path: str = "data/BitcoinHeistData.csv" # Raw upstream data.
    preprocessing_data_path: str = "data/intermediate/preprocessing/"
    features_data_path: str = "data/intermediate/features/"
    raw_csv_path: str = "data/BitcoinHeistData.csv" 

    model_path: str = "models/"  # Model store


@dataclass(init=False, frozen=True)
class RunConfig:
    random_seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.2 
    num_folds: int = 5
    evaluate_model: bool = True
    log_to_mlflow: bool = False
    mlflow_tracking_uri: str = "http://mlflow:8080"
    experiment_name: str = "Alpha"


@dataclass(init=False, frozen=True)
class ModelConfig:
    model_name: str = "alpha"
    n_estimators: int = 100
    max_depth: int = 100
    random_state: int = 42


@dataclass(init=False, frozen=True)
class DatabaseConfig:
    host: str = os.getenv("MYSQL_HOST", "localhost")
    port: int = 3306
    database: str = "transactions_db"
    user: str = "user"
    password: str = "password"
    raw_table: str = "raw_transactions"
    preprocessing_table: str = "preprocessed_data"
    features_table: str = "features_data"