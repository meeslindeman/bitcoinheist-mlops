from pyspark.sql import SparkSession, DataFrame
from configs.configs import PathsConfig


def get_spark_session(app_name: str, master: str = "local[2]") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .getOrCreate()
    )


def read_raw_data() -> DataFrame:
    spark = get_spark_session(app_name="bitcoin-heist-raw-data")
    return spark.read.parquet(PathsConfig.raw_data_path)


def write_data(df: DataFrame, path: str, mode: str = "overwrite") -> None:
    df.write.mode(mode).parquet(path)


def read_data(path: str) -> DataFrame:
    spark = get_spark_session(app_name="bitcoin-heist-read-data")
    return spark.read.parquet(path)