from pyspark.sql import SparkSession

from configs.configs import PathsConfig, RunConfig
from src.features import get_features
from src.data_preprocessing import data_preprocessing
from src.model_training import Model

def model_training(data) -> None:
    model = Model()
    model.train_model(data)

    print("\nCross-validation scores:")
    print(model.get_cv_scores(data))

    if RunConfig.evaluate_model:
        print("\nEvaluating model on test set:")
        model.evaluate_model(data)

    model.save_model()

    if RunConfig.log_to_mlflow:
        model.log_modelto_mlflow()

def main():
    spark = SparkSession.builder.appName("RansomwareDetection").getOrCreate()
    data = (spark.read.csv(PathsConfig.raw_csv_path, header=True, inferSchema=True))

    data = data_preprocessing(data)
    data = get_features(data)

    data_pd = data.toPandas()

    model_training(data_pd)

    spark.stop()

if __name__ == "__main__":
    main()