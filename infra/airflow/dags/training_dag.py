from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

# This network name must match the compose project network name.
# If your compose file is infra/docker-compose.yaml, the default is "infra_default".
DOCKER_NETWORK = "infra_default"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="bitcoin_heist_training",
    default_args=default_args,
    description="Bitcoin Heist: CSV -> Parquet -> features -> model (with MLflow logging)",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["bitcoin_heist", "training"],
) as dag:

    init_parquet = DockerOperator(
        task_id="init_parquet",
        image="bitcoinheist-app:latest",
        command="python scripts/csv_to_parquet.py",
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        auto_remove=True,
        mount_tmp_dir=False,
    )

    train_model = DockerOperator(
        task_id="train_model",
        image="bitcoinheist-app:latest",
        command="python src/main_training.py --preprocess --feat-eng --training",
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        auto_remove=True,
        mount_tmp_dir=False,
    )

    init_parquet >> train_model