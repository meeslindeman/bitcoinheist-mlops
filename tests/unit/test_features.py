import math

import pandas as pd
import pytest
from pyspark.testing.utils import assertDataFrameEqual

from src.features import *

def test_get_log_transformed_features(spark_session):
    data = spark_session.createDataFrame(
        [
            {"income": 0.0, "weight": 10.0, "count": 2.0, "looped": 1.0},
            {"income": 100.0, "weight": 5.0, "count": 0.0, "looped": 0.0},
        ]
    )

    out = get_log_transformed_features(
        data,
        ["income", "weight", "count", "looped"],
    )

    # note: columns created
    for col in ["log_income", "log_weight", "log_count", "log_looped"]:
        assert col in out.columns

    # note: check a few concrete values with approx
    row0 = out.collect()[0]
    assert row0["log_income"] == pytest.approx(math.log1p(0.0))
    assert row0["log_weight"] == pytest.approx(math.log1p(10.0))
    assert row0["log_count"] == pytest.approx(math.log1p(2.0))
    assert row0["log_looped"] == pytest.approx(math.log1p(1.0))


def test_get_ratio_features(spark_session):
    data = spark_session.createDataFrame(
        [
            {"count": 0.0, "neighbors": 0.0, "looped": 0.0, "weight": 10.0, "income": 100.0},
            {"count": 1.0, "neighbors": 1.0, "looped": 1.0, "weight": 20.0, "income": 200.0},
        ]
    )

    out = get_ratio_features(data)

    expected = spark_session.createDataFrame(
        [
            {
                "count": 0.0,
                "neighbors": 0.0,
                "looped": 0.0,
                "weight": 10.0,
                "income": 100.0,
                "activity_index": 0.0,          # (0 + 0) / (0 + 1)
                "weight_to_count": 10.0,        # 10 / (0 + 1)
                "income_per_neighbor": 100.0,   # 100 / (0 + 1)
            },
            {
                "count": 1.0,
                "neighbors": 1.0,
                "looped": 1.0,
                "weight": 20.0,
                "income": 200.0,
                "activity_index": 1.0,          # (1 + 1) / (1 + 1)
                "weight_to_count": 10.0,        # 20 / (1 + 1)
                "income_per_neighbor": 100.0,   # 200 / (1 + 1)
            },
        ]
    )

    # note: compare including existing columns
    assertDataFrameEqual(
        out.select(*expected.columns),
        expected,
        ignoreColumnOrder=True,
    )


def test_get_temporal_features(spark_session):
    data = spark_session.createDataFrame(
        [
            {"year": 2009, "day": 1},
            {"year": 2010, "day": 365},
        ]
    )

    out = get_temporal_features(data)

    rows = out.collect()
    row0, row1 = rows[0], rows[1]

    # note: (2009 - 2009) * 365 + 1 = 1
    assert row0["days_since_2009"] == 1
    # note: (2010 - 2009) * 365 + 365 = 730
    assert row1["days_since_2009"] == 730


def test_get_features(spark_session):
    data = spark_session.createDataFrame(
        [
            {
                "year": 2010,
                "day": 10,
                "length": 5.0,
                "weight": 20.0,
                "count": 2.0,
                "looped": 1.0,
                "neighbors": 3.0,
                "income": 50.0,
                "is_ransomware": 1,
            },
            {
                "year": 2011,
                "day": 100,
                "length": 8.0,
                "weight": 5.0,
                "count": 0.0,
                "looped": 0.0,
                "neighbors": 0.0,
                "income": 0.0,
                "is_ransomware": 0,
            },
        ]
    )

    out = get_features(data)

    expected_cols = {
        "day",
        "weight",
        "count",
        "looped",
        "neighbors",
        "is_ransomware",
        "log_income",
        "log_weight",
        "log_count",
        "log_looped",
        "activity_index",
        "weight_to_count",
        "income_per_neighbor",
        "days_since_2009",
        "sin_day",
        "cos_day",
    }

    assert expected_cols.issubset(set(out.columns))