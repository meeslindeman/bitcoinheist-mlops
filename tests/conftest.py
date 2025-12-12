import pytest
import src.app.main_api as main_api


# note: reuse the spark session fixture in main_api to avoid creating multiple Spark sessions
@pytest.fixture(autouse=True)
def reuse_spark_session(spark_fixture):
    main_api.spark = spark_fixture
    yield
    main_api.spark = None

