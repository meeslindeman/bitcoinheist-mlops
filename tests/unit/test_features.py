import math

import pandas as pd
import pytest
from pyspark.testing.utils import assertDataFrameEqual

from src.features import (
    get_log_transformed_features,
    get_ratio_features,
    get_temporal_features,
    _get_z_score,
    get_features,
)


def test_get_log_transformed_features(spark_fixture):
    data = spark_fixture.createDataFrame(
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


def test_get_log_transformed_features_boundaries(spark_fixture):
    data = spark_fixture.createDataFrame(
        [
            {"income": 0.0, "weight": 0.0, "count": 0.0, "looped": 0.0},
            {"income": 0.5, "weight": 10.0, "count": 1.0, "looped": 2.0},
            {"income": -0.5, "weight": 1.0, "count": 0.0, "looped": 0.0},
        ]
    )

    out = get_log_transformed_features(
        data,
        ["income", "weight", "count", "looped"],
    )
    rows = out.collect()

    # note: 0 -> log1p(0) = 0
    assert rows[0]["log_income"] == pytest.approx(0.0)
    assert rows[1]["log_income"] == pytest.approx(math.log1p(0.5))
    assert rows[2]["log_income"] == pytest.approx(math.log1p(-0.5))


def test_get_ratio_features(spark_fixture):
    data = spark_fixture.createDataFrame(
        [
            {"count": 0.0, "neighbors": 0.0, "looped": 0.0, "weight": 10.0, "income": 100.0},
            {"count": 1.0, "neighbors": 1.0, "looped": 1.0, "weight": 20.0, "income": 200.0},
        ]
    )

    out = get_ratio_features(data)

    expected = spark_fixture.createDataFrame(
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


def test_get_ratio_features_boundaries(spark_fixture):
    data = spark_fixture.createDataFrame(
        [
            {
                "count": 0.0,
                "neighbors": 0.0,
                "looped": 0.0,
                "weight": 0.0,
                "income": 0.0,
            },
            {
                "count": 1e6,
                "neighbors": 1e6,
                "looped": 1e6,
                "weight": 1e6,
                "income": 1e6,
            },
        ]
    )

    out = get_ratio_features(data)
    rows = out.collect()

    # note: row 0: denominators are +1
    r0 = rows[0]
    assert r0["activity_index"] == pytest.approx(0.0)  # (0+0)/(0+1)
    assert r0["weight_to_count"] == pytest.approx(0.0)  # 0/(0+1)
    assert r0["income_per_neighbor"] == pytest.approx(0.0)  # 0/(0+1)

    # note: row 1: sanity check non-zero finite outputs
    r1 = rows[1]
    assert math.isfinite(r1["activity_index"])
    assert math.isfinite(r1["weight_to_count"])
    assert math.isfinite(r1["income_per_neighbor"])


def test_get_temporal_features(spark_fixture):
    data = spark_fixture.createDataFrame(
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


def test_get_temporal_features_extended(spark_fixture):
    data = spark_fixture.createDataFrame(
        [
            {"year": 2010, "day": 0},
            {"year": 2010, "day": 3650},  # 10 years worth of days
        ]
    )

    out = get_temporal_features(data)
    rows = out.collect()

    for r in rows:
        assert math.isfinite(r["sin_day"])
        assert math.isfinite(r["cos_day"])

    # note: days_since_2009 still linear in year/day
    assert rows[0]["days_since_2009"] == (2010 - 2009) * 365 + 0
    assert rows[1]["days_since_2009"] == (2010 - 2009) * 365 + 3650


def test_get_z_score_global(spark_fixture):
    data = spark_fixture.createDataFrame(
        [
            {"income": 1.0},
            {"income": 2.0},
            {"income": 3.0},
        ]
    )

    out = _get_z_score(data, value_col="income")

    # note: mean = 2, std = 1, so z = income - 2
    expected = spark_fixture.createDataFrame(
        [
            {"income": 1.0, "z_income": -1.0},
            {"income": 2.0, "z_income": 0.0},
            {"income": 3.0, "z_income": 1.0},
        ]
    )

    assertDataFrameEqual(
        out.select("income", "z_income"),
        expected,
        ignoreColumnOrder=True,
    )


def test_get_z_score_grouped(spark_fixture):
    data = spark_fixture.createDataFrame(
        [
            {"income": 1.0, "year": 2010},
            {"income": 3.0, "year": 2010},
            {"income": 10.0, "year": 2011},
            {"income": 10.0, "year": 2011},
        ]
    )

    out = _get_z_score(data, value_col="income", group_by_cols=["year"])

    rows = { (r["year"], r["income"]): r["z_income_by_year"] for r in out.collect() }

    # note: year 2010: incomes [1,3] -> mean=2, std= sqrt(((1-2)^2+(3-2)^2)/(n-1)) = sqrt(2)
    assert rows[(2010, 1.0)] == pytest.approx((1.0 - 2.0) / (2.0 ** 0.5))
    assert rows[(2010, 3.0)] == pytest.approx((3.0 - 2.0) / (2.0 ** 0.5))

    # note: year 2011: incomes [10,10] -> std=0 -> z should be 0.0 by design
    assert rows[(2011, 10.0)] == pytest.approx(0.0)


def test_get_z_score_with_nulls(spark_fixture):
    data = spark_fixture.createDataFrame(
        [
            {"income": 10.0},
            {"income": None},
            {"income": 20.0},
        ]
    )

    out = _get_z_score(data, value_col="income")
    rows = {r["income"]: r["z_income"] for r in out.collect()}

    assert set(rows.keys()) == {10.0, 20.0, 0.0}

    # note: all z-scores should be finite
    for z in rows.values():
        assert math.isfinite(z)


def test_get_features(spark_fixture):
    data = spark_fixture.createDataFrame(
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
        "z_income",
        "z_income_by_year"
    }

    assert expected_cols.issubset(set(out.columns))