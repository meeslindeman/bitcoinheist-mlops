import pandas as pd
from pyspark.sql import SparkSession

from configs.configs import PathsConfig

def get_spark_session():
    spark = (
        SparkSession.builder
        .appName("bitcoin-heist-csv-to-parquet")
        .master("local[2]")
        .getOrCreate()
    )
    return spark

def csv_to_parquet() -> None:
    raw_csv = pd.read_csv(PathsConfig.raw_csv_path)

    # Optional: drop common auto-generated index column if present
    if "Unnamed: 0" in raw_csv.columns:
        raw_csv = raw_csv.drop(columns=["Unnamed: 0"])

    # Here you can add dataset-specific cleaning for the Bitcoin Heist dataset
    # For example:
    #   - type conversions
    #   - handling missing values
    #   - normalizing label columns
    #
    # Keep it minimal at this step; feature engineering will follow later.

    spark = get_spark_session()
    spark_df = spark.createDataFrame(raw_csv)

    # If you have a natural partition column, you can partition by it.
    # For now, write without partitioning to keep it simple:
    spark_df.write.mode("overwrite").parquet(PathsConfig.raw_data_path)


if __name__ == "__main__":
    csv_to_parquet()

