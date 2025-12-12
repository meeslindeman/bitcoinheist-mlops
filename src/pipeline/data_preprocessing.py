from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from configs.configs import RunConfig


def data_preprocessing(data: DataFrame) -> DataFrame:
    # note: drop 'address' column as it is not useful for modeling
    if "address" in data.columns:
        data = data.drop("address")

    # note: dataset has no nans but we fill numeric columns with 0.0 for safety
    numeric_cols = ["year", "day", "length", "weight", "count", "looped", "neighbors", "income"]
    data = data.fillna({col: 0.0 for col in numeric_cols})

    # note: make it a binary classification problem
    ransom_df = data.filter(F.col("label") != "white")
    white_df = data.filter(F.col("label") == "white")

    # note: counts to balance classes
    ransom_count = ransom_df.count()
    white_count = white_df.count()
    
    # note: fraction of white to sample to match ransom count
    if white_count == 0:
        raise ValueError("No 'white' class instances found in the dataset.")
    
    # note: downsample the majority class to balance the dataset
    white_df = (white_df.orderBy(F.rand(seed=RunConfig.random_seed)).limit(ransom_count))
    data = ransom_df.unionByName(white_df)

    # note: assert balanced classes
    white_balanced = data.filter(F.col("label") == "white").count()
    ransom_balanced = data.filter(F.col("label") != "white").count()
    assert white_balanced == ransom_balanced, "Classes are not balanced after preprocessing."

    # note: convert label to binary
    data = data.withColumn("is_ransomware", (F.col("label") != "white").cast("int")).drop("label")

    return data