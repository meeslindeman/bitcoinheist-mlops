from pyspark.sql import SparkSession, DataFrame
from configs.configs import PathsConfig


def get_spark_session() -> SparkSession:
    """
    Create or retrieve a SparkSession.
    Restrict cores to avoid using the full machine.
    """
    spark = (
        SparkSession.builder
        .appName("bitcoin-heist-pipeline")
        .master("local[2]")
        .getOrCreate()
    )
    return spark


def read_raw_data() -> DataFrame:
    """
    Read raw data from Parquet.
    Used in the preprocessing step instead of reading from MySQL.
    """
    spark = get_spark_session()
    return spark.read.parquet(PathsConfig.raw_data_path)


def write_preprocessed_data(df: DataFrame, mode: str = "overwrite") -> None:
    """
    Write preprocessed data to Parquet.
    Replacement for preprocessed_table in MySQL.
    """
    df.write.mode(mode).parquet(PathsConfig.preprocessed_data_path)


def read_preprocessed_data() -> DataFrame:
    """
    Read preprocessed data from Parquet.
    Used in the feature engineering step instead of reading from MySQL.
    """
    spark = get_spark_session()
    return spark.read.parquet(PathsConfig.preprocessed_data_path)


def write_features_data(df: DataFrame, mode: str = "overwrite") -> None:
    """
    Write feature-engineered data to Parquet.
    Replacement for features_table in MySQL.
    """
    df.write.mode(mode).parquet(PathsConfig.features_data_path)


def read_features_data() -> DataFrame:
    """
    Read feature-engineered data from Parquet.
    Used in the training step instead of reading from MySQL.
    """
    spark = get_spark_session()
    return spark.read.parquet(PathsConfig.features_data_path)
