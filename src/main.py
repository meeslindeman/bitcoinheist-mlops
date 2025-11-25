from pyspark.sql import SparkSession

from configs.configs import DatabaseConfig, RunConfig
from src.data_loader import read_from_sql, write_to_sql
from src.data_preprocessing import data_preprocessing
from src.features import get_features
from src.model_training import Model


def main():
    # note: load raw data from MySQL
    raw_data = read_from_sql(f"SELECT * FROM {DatabaseConfig.raw_table}")
    
    # note: preprocess data
    preprocessed_data = data_preprocessing(raw_data)
    write_to_sql(
        preprocessed_data, 
        DatabaseConfig.preprocessed_table, 
        mode='overwrite'
    )

    # note: extract features
    features_data = get_features(preprocessed_data)
    write_to_sql(
        features_data,
        DatabaseConfig.features_table,
        mode="overwrite",
    )

    # note: train model
    features_pd = features_data.toPandas()
    model = Model()

    print("\nCross-validation scores:")
    print(model.get_cv_scores(features_pd))

    model.train_model(features_pd)

    if getattr(RunConfig, "evaluate_model", False):
        print("\nEvaluating model on test set:")
        model.evaluate_model(features_pd)

    model.save_model()

    if RunConfig.log_to_mlflow:
        model.log_model_to_mlflow(
            features_pd,
            tracking_uri=RunConfig.mlflow_tracking_uri,
            experiment_name=RunConfig.experiment_name,
        )

if __name__ == "__main__":
    main()