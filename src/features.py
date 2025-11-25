from typing import List
import numpy as np

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

def get_log_transformed_features(data: DataFrame, features: List[str]) -> DataFrame:
    for feature in features:
        log_feature = f"log_{feature}"
        data = data.withColumn(log_feature, F.log1p(F.col(feature)))
    return data


def get_ratio_features(data: DataFrame) -> DataFrame:
    # note: (count + neighbors) / (looped + 1)
    data = data.withColumn(
        "activity index",
        (F.col("count") + F.col("neighbors")) / (F.col("looped") + F.lit(1))
    )

    # note: weight / (count + 1)
    data = data.withColumn(
        "weight_to_count",
        F.col("weight") / (F.col("count") + F.lit(1))
    )

    # note: income / (neighbors + 1)
    data = data.withColumn(
        "income_per_neighbor",
        F.col("income") / (F.col("neighbors") + F.lit(1)),
    )

    return data


def get_temporal_features(data: DataFrame) -> DataFrame:
    # note: (year - 2009) * 365 + day
    data = data.withColumn(
        "days_since_2009",
        (F.col("year") - F.lit(2009)) * F.lit(365) + F.col("day"),
    )

    # note: using 365 day period: 2 * pi * day / 365
    angle = 2 * F.lit(np.pi) * F.col("day") / F.lit(365)

    data = data.withColumn("sin_day", F.sin(angle))
    data = data.withColumn("cos_day", F.cos(angle))

    return data


def _get_z_score(data: DataFrame, value_col: str, group_by_cols: List[str] | None = None) -> DataFrame:
    group_by_cols = group_by_cols or []

    stats = (data.groupBy(group_by_cols) if group_by_cols else data.groupBy()).agg(
        F.mean(value_col).alias("mean"),
        F.stddev(value_col).alias("std"),
    )

    if group_by_cols:
        data = data.join(stats, on=group_by_cols)
    else:
        # note: cross join to attach same mean/std to all rows
        data = data.crossJoin(stats)

    if group_by_cols:
        z_col_name = f"z_{value_col}_by_{'_'.join(group_by_cols)}"
    else:
        z_col_name = f"z_{value_col}"

    data = data.withColumn(
        z_col_name,
        F.when(F.col("std").isNull() | (F.col("std") == 0), F.lit(0.0)).otherwise(
            (F.col(value_col) - F.col("mean")) / F.col("std")
        ),
    )

    data = data.drop("mean", "std")
    return data


def get_features(data: DataFrame) -> DataFrame:
    data = get_log_transformed_features(
        data, 
        ['income', 'weight', 'count', 'looped']
    )
    data = get_ratio_features(data)

    # note: here we do z-score on income globally and within each year
    data = _get_z_score(data, value_col="income")
    data = _get_z_score(data, value_col="income", group_by_cols=["year"])

    data = get_temporal_features(data)

    # note: drop original columns that are not needed anymore
    drop_cols = ["year", "income", "length"]
    data = data.drop(*drop_cols)

    print(f"Final feature set columns: {data.columns}")
    return data