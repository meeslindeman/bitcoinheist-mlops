import pandas as pd
from configs.configs import PathsConfig, DatabaseConfig
from src.data_loader import get_sql_engine

def load_csv_to_mysql(file_path: str, table_name: str, mode: str = 'overwrite') -> None:
    """Load data from a CSV file into a MySQL table."""
    pandas_df = pd.read_csv(file_path)
    
    engine = get_sql_engine()
    
    if_exists_option = 'replace' if mode == 'overwrite' else 'append'
    pandas_df.to_sql(table_name, con=engine, if_exists=if_exists_option, index=False)

    print(f"Data loaded into table '{table_name}' from '{file_path}' with mode '{mode}'.")

if __name__ == "__main__":
    load_csv_to_mysql(
        file_path=PathsConfig.raw_csv_path,
        table_name=DatabaseConfig.raw_table,
        mode='overwrite'
    )