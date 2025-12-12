import json
import logging
import click

from configs.configs import PathsConfig, RunConfig, TelemetryConfig
from src.utils.spark_utils import read_raw_data, write_data, read_data
from src.pipeline.data_preprocessing import data_preprocessing
from src.pipeline.features import get_features
from src.pipeline.feature_extractor import FeatureExtractor
from src.model.model import Model
from src.telemetry.training import push_training_summary
from src.telemetry.training_data import save_training_distribution


logger = logging.getLogger(__name__)


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
        logger.info(f"[Preprocess] Wrote preprocessed data to: {PathsConfig.preprocessed_data_path}")

    if feat_eng:
        # note: stateless Spark feature engineering: log/ratio/temporal
        preprocessed_data = read_data(path=PathsConfig.preprocessed_data_path)
        features_data = get_features(preprocessed_data)

        # note: stateful feature extraction: z-score normalization
        extractor = FeatureExtractor()
        features_data = extractor.get_features(features_data)

        # note: drop unused columns, engineered features work better
        drop_cols = ["year", "income", "length"]
        existing_drop_cols = [c for c in drop_cols if c in features_data.columns]
        if existing_drop_cols:
            features_data = features_data.drop(*existing_drop_cols)

        # note: save engineered features to parquet
        write_data(features_data, path=PathsConfig.features_data_path, mode="overwrite")
        logger.info(f"[Feat-Eng] Wrote features data to: {PathsConfig.features_data_path}")

        # note: save the z-score state so we can reuse it for inference
        extractor.save_state(PathsConfig.feature_extractor_state_dir)
        logger.info(f"[Feat-Eng] Saved FeatureExtractor state to: {PathsConfig.feature_extractor_state_dir}")

    if training:
        features_spark = read_data(path=PathsConfig.features_data_path)
        # note: convert to Pandas for model training
        features_data = features_spark.toPandas()

        # note: ensure target column exists
        if "is_ransomware" not in features_data.columns:
            raise ValueError("Column 'is_ransomware' not found in features data for training.")

        # note: ensures inference uses the same feature set and order as training
        # note: mismatch in feature columns leads to incorrect model predictions or runtime errors
        # note: more robust than hardcoding feature columns in \predict api
        feature_columns = [c for c in features_data.columns if c != "is_ransomware"]
        with open(PathsConfig.feature_columns_path, "w") as f:
            json.dump(feature_columns, f)

        # note: model training
        model = Model()
        model.train_model(features_data)

        logger.info(f"[Training] Trained model and saved to: {PathsConfig.trained_model_path}")
        logger.info(model.get_cv_scores())

        if RunConfig.evaluate_model:
            logger.info("[Training] Test metrics:")
            model.evaluate_model(features_data)

        # note: write training telemetry metrics
        test_summary = model.get_test_summary()
        push_training_summary(test_summary)
        logger.info("[Training] Pushed training summary metrics to Pushgateway")

        # note: save preprocessed data distribution for data drift monitoring
        preprocessed_data = read_data(path=PathsConfig.preprocessed_data_path)
        preprocessed_df = preprocessed_data.toPandas()
        save_training_distribution(preprocessed_df)
        logger.info(f"[Training] Wrote training data distribution to: {TelemetryConfig.telemetry_training_data_dist_path}")

        # note: log to MLflow
        if RunConfig.log_to_mlflow:
            logged_run_id = model.log_model_to_mlflow()
            logger.info(f"[Training] Logged model to MLflow run: {logged_run_id}")


if __name__ == "__main__":
    main()