import pandas as pd

from configs.configs import PathsConfig
from src.spark_utils import get_spark_session

def csv_to_parquet() -> None:
    raw_csv = pd.read_csv(PathsConfig.raw_csv_path)

    if "Unnamed: 0" in raw_csv.columns:
        raw_csv = raw_csv.drop(columns=["Unnamed: 0"])

    spark = get_spark_session(app_name="csv-to-parquet")
    spark_df = spark.createDataFrame(raw_csv)
    spark_df.write.mode("overwrite").parquet(PathsConfig.raw_data_path)

if __name__ == "__main__":
    csv_to_parquet()