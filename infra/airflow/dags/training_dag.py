from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    "start_date": datetime(2025, 1, 1),
}

dag = DAG(
    dag_id="bitcoin-heist-training",
    default_args=default_args
)

log_datetime_start_task = BashOperator(
    task_id="log_datetime_start", 
    bash_command="date", 
    dag=dag
)

mounts = [
    Mount(source="bitcoin-heist-data", target="/bitcoinheist-app/data", type="volume"),
    Mount(source="bitcoin-heist-models", target="/bitcoinheist-app/models", type="volume"),
    Mount(source="bitcoin-heist-telemetry", target="/bitcoinheist-app/telemetry", type="volume")
]

init_parquet_task = DockerOperator(
    task_id="init_parquet",
    docker_url="unix://var/run/docker.sock",
    image="bitcoinheist-app:latest",
    command="python scripts/csv_to_parquet.py",
    network_mode="infra_default",
    mounts=mounts,
    dag=dag
)

preprocessing_task = DockerOperator(
    task_id="data_preprocessing",
    docker_url="unix://var/run/docker.sock",
    image="bitcoinheist-app:latest",
    command="python src/main_training.py --preprocess",
    network_mode="infra_default",
    mounts=mounts,
    dag=dag
)

feat_eng_task = DockerOperator(
    task_id="feature_engineering",
    docker_url="unix://var/run/docker.sock",
    image="bitcoinheist-app:latest",
    command="python src/main_training.py --feat-eng",
    network_mode="infra_default",
    mounts=mounts,
    dag=dag
)

model_training_task = DockerOperator(
    task_id="training",
    docker_url="unix://var/run/docker.sock",
    image="bitcoinheist-app:latest",
    command="python src/main_training.py --training",
    network_mode="infra_default",
    mounts=mounts,
    dag=dag,
)

reload_task = BashOperator(
    task_id="reload_api_model",
    bash_command="curl -X POST http://app:5001/reload",
    dag=dag
)

log_datetime_end_task = BashOperator(
    task_id="log_datetime_end", 
    bash_command="date", 
    dag=dag
)

(
    log_datetime_start_task
    >> init_parquet_task
    >> preprocessing_task
    >> feat_eng_task
    >> model_training_task
    >> reload_task
    >> log_datetime_end_task
)