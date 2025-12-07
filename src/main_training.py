import json
import click

from configs.configs import (
    PathsConfig,
    RunConfig
)
from src.spark_utils import (
    read_raw_data,
    write_data,
    read_data,
)
from src.data_preprocessing import data_preprocessing
from src.features import get_features
from src.model import Model
from src.telemetry.training import write_training_summary


@click.command()
@click.option("--preprocess", is_flag=True, help="Run preprocessing step (raw → balanced preprocessed).")
@click.option("--feat-eng", is_flag=True, help="Run feature engineering step (preprocessed → features).")
@click.option("--training", is_flag=True, help="Run model training step (features → model + metrics).")
def main(preprocess: bool, feat_eng: bool, training: bool):

    if preprocess:
        # note: raw csv to preprocessed parquet
        raw_data = read_raw_data()
        preprocessed_data = data_preprocessing(raw_data)
        write_data(preprocessed_data, path=PathsConfig.preprocessed_data_path, mode="overwrite")
        print(f"[preprocess] Wrote preprocessed data to: {PathsConfig.preprocessed_data_path}")

    if feat_eng:
        # note: preprocessed parquet to features parquet
        preprocessed_data = read_data(path=PathsConfig.preprocessed_data_path)
        features_data = get_features(preprocessed_data)

        write_data(features_data, path=PathsConfig.features_data_path, mode="overwrite")
        print(f"[feat-eng] Wrote features data to: {PathsConfig.features_data_path}")

    if training:
        features_spark = read_data(path=PathsConfig.features_data_path)
        features_data = features_spark.toPandas()

        # note: ensure target column exists
        if "is_ransomware" not in features_data.columns:
            raise ValueError("Column 'is_ransomware' not found in features data for training.")

        # note: persist feature column order for inference
        feature_columns = [c for c in features_data.columns if c != "is_ransomware"]
        with open(PathsConfig.feature_columns_path, "w") as f:
            json.dump(feature_columns, f)

        # note: model training
        model = Model()
        model.train_model(features_data)

        print("\n[training] Cross-validation scores:")
        print(model.get_cv_scores())

        if RunConfig.evaluate_model:
            print("\n[training] Test metrics:")
            model.evaluate_model(features_data)

        model.save_model_local()

        # note: write training telemetry
        test_summary = model.get_test_summary()
        write_training_summary(test_summary)
        print(f"[training] Wrote training summary telemetry to: {PathsConfig.telemetry_training_data_path}")

        # note: log to MLflow
        if RunConfig.log_to_mlflow:
            logged_run_id = model.log_model_to_mlflow()
            print(f"[training] Logged model to MLflow run: {logged_run_id}")

if __name__ == "__main__":
    main()
