import json
import datetime
import os

import click
import mlflow
import pyspark.sql.functions as F

from configs.configs import (
    PathsConfig,
    RunConfig,
    ModelConfig,
)
from src.data_loader import (
    read_raw_data,
    write_preprocessed_data,
    read_preprocessed_data,
    write_features_data,
    read_features_data,
)
from src.data_preprocessing import data_preprocessing
from src.feature_extractor import FeatureExtractor
from src.model import Model


@click.command()
@click.option("--preprocess", is_flag=True, help="Run preprocessing step (raw → balanced preprocessed).")
@click.option("--feat-eng", is_flag=True, help="Run feature engineering step (preprocessed → features).")
@click.option("--training", is_flag=True, help="Run model training step (features → model + metrics).")
def main(preprocess: bool, feat_eng: bool, training: bool):

    if preprocess:
        # 1. raw parquet → preprocessed parquet
        raw_data = read_raw_data()
        preprocessed_data = data_preprocessing(raw_data)
        write_preprocessed_data(preprocessed_data, mode="overwrite")
        print(f"[preprocess] Wrote preprocessed data to: {PathsConfig.preprocessed_data_path}")

    if feat_eng:
        # 2. preprocessed parquet → features parquet
        preprocessed_data = read_preprocessed_data()

        feature_extractor = FeatureExtractor()
        features_data = feature_extractor.get_features(preprocessed_data)

        write_features_data(features_data, mode="overwrite")
        print(f"[feat-eng] Wrote features data to: {PathsConfig.features_data_path}")

    if training:
        features_spark = read_features_data()
        features_data = features_spark.toPandas()

        # ensure target column exists
        if "is_ransomware" not in features_data.columns:
            raise ValueError("Column 'is_ransomware' not found in features data for training.")

        # persist feature column order for inference
        feature_columns = [c for c in features_data.columns if c != "is_ransomware"]
        with open(PathsConfig.feature_columns_path, "w") as f:
            json.dump(feature_columns, f)

        # train model on full dataframe (Model will split internally)
        model = Model()
        model.train_model(features_data)

        print("\n[training] Cross-validation scores:")
        print(model.get_cv_scores())

        if RunConfig.evaluate_model:
            print("\n[training] Test metrics:")
            model.evaluate_model(features_data)

        # local save
        model.save_model_local()

        # MLflow logging (unchanged)
        if RunConfig.log_to_mlflow:
            logged_run_id = model.log_model_to_mlflow()
            print(f"[training] Logged model to MLflow run: {logged_run_id}")

        telemetry_data = {
            "is_ransomware": {
                0: int((features_data["is_ransomware"] == 0).sum()),
                1: int((features_data["is_ransomware"] == 1).sum()),
            }
        }

        telemetry_path = getattr(PathsConfig, "telemetry_training_data_path", None)
        if telemetry_path is not None:
            telemetry_dir = os.path.dirname(telemetry_path)
            os.makedirs(telemetry_dir, exist_ok=True)

            with open(telemetry_path, "w") as f:
                json.dump(telemetry_data, f)
            print(f"[training] Wrote telemetry summary to: {telemetry_path}")


if __name__ == "__main__":
    main()
