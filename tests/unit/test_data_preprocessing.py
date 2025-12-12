import pyspark.sql.functions as F

from src.pipeline.data_preprocessing import data_preprocessing


def test_data_preprocessing(spark_fixture):
    # note: should be 3 ransom and 3 white after balancing
    data = spark_fixture.createDataFrame(
        [
            {
                "address": "a1",
                "year": 2010,
                "day": 1,
                "length": 5.0,
                "weight": 10.0,
                "count": 1.0,
                "looped": 0.0,
                "neighbors": 2.0,
                "income": 100.0,
                "label": "white",
            },
            {
                "address": "a2",
                "year": 2010,
                "day": 2,
                "length": 5.0,
                "weight": 20.0,
                "count": 2.0,
                "looped": 1.0,
                "neighbors": 3.0,
                "income": 200.0,
                "label": "white",
            },
            {
                "address": "a3",
                "year": 2010,
                "day": 3,
                "length": 5.0,
                "weight": 30.0,
                "count": 3.0,
                "looped": 1.0,
                "neighbors": 4.0,
                "income": 300.0,
                "label": "white",
            },
            {
                "address": "a4",
                "year": 2010,
                "day": 4,
                "length": 5.0,
                "weight": 40.0,
                "count": 4.0,
                "looped": 2.0,
                "neighbors": 5.0,
                "income": 400.0,
                "label": "white",
            },
            {
                "address": "r1",
                "year": 2010,
                "day": 5,
                "length": 5.0,
                "weight": 50.0,
                "count": 5.0,
                "looped": 2.0,
                "neighbors": 6.0,
                "income": 500.0,
                "label": "ransomware_family_1",
            },
            {
                "address": "r2",
                "year": 2010,
                "day": 6,
                "length": 5.0,
                "weight": 60.0,
                "count": 6.0,
                "looped": 3.0,
                "neighbors": 7.0,
                "income": 600.0,
                "label": "ransomware_family_2",
            },
            {
                "address": "r3",
                "year": 2010,
                "day": 7,
                "length": 5.0,
                "weight": 70.0,
                "count": 7.0,
                "looped": 3.0,
                "neighbors": 8.0,
                "income": 700.0,
                "label": "ransomware_family_3",
            },
        ]
    )

    out = data_preprocessing(data)

    # note: label and address should be dropped
    assert "label" not in out.columns
    assert "address" not in out.columns

    # note: is_ransomware should be present and binary
    assert "is_ransomware" in out.columns
    is_ransomware_values = [row["is_ransomware"] for row in out.select("is_ransomware").distinct().collect()]
    assert set(is_ransomware_values) == {0, 1}

    # note: classes should be balanced
    ransom_count = out.filter(F.col("is_ransomware") == 1).count()
    white_count = out.filter(F.col("is_ransomware") == 0).count()
    assert ransom_count == white_count == 3