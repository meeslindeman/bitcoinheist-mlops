import pandas as pd
from sqlalchemy import create_engine, text
from pyspark.sql import SparkSession, DataFrame
from configs.configs import DatabaseConfig

def get_sql_engine():
    """Create and return a SQLAlchemy engine for MySQL connection."""
    engine = create_engine(
        f"mysql+pymysql://{DatabaseConfig.user}:{DatabaseConfig.password}@{DatabaseConfig.host}:{DatabaseConfig.port}/{DatabaseConfig.database}"
    )
    return engine

def read_from_sql(query: str) -> DataFrame:
    """MySQL Table → SQL Query → Pandas DataFrame → PySpark DataFrame"""
    engine = get_sql_engine()
    with engine.connect() as connection:
        result = pd.read_sql(text(query), connection)
    spark = SparkSession.builder.master("local[2]").getOrCreate()
    result = spark.createDataFrame(result)
    return result

def write_to_sql(df: DataFrame, table_name: str, mode: str = 'overwrite') -> None:
    """PySpark DataFrame → Pandas DataFrame → MySQL Table"""
    engine = get_sql_engine()
    
    pandas_df = df.toPandas()
    
    if_exists_option = 'replace' if mode == 'overwrite' else 'append'
    pandas_df.to_sql(table_name, con=engine, if_exists=if_exists_option, index=False)
    