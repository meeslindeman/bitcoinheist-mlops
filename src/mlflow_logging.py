import os
import pickle
import tempfile
from typing import Optional

import mlflow
from sklearn.base import BaseEstimator

from configs.configs import ModelConfig, RunConfig


def _get_or_create_experiment_id() -> str:
    mlflow.set_tracking_uri(RunConfig.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(RunConfig.experiment_name)
    if experiment is None:
        return mlflow.create_experiment(RunConfig.experiment_name)
    return experiment.experiment_id


def log_model_to_mlflow(model: BaseEstimator, cv_scores: dict, test_report: Optional[dict], test_roc_auc: Optional[float]) -> str:
    experiment_id = _get_or_create_experiment_id()

    with mlflow.start_run(experiment_id=experiment_id, run_name=RunConfig.run_name) as run:
        run_id = run.info.run_id

        mlflow.log_dict(cv_scores, "cv_scores.json")
        if "test_f1" in cv_scores:
            mlflow.log_metric("cv_test_f1_mean", cv_scores["test_f1"].mean())
            mlflow.log_metric("cv_test_f1_std", cv_scores["test_f1"].std())
        if "test_precision" in cv_scores:
            mlflow.log_metric(
                "cv_test_precision_mean", cv_scores["test_precision"].mean()
            )
        if "test_recall" in cv_scores:
            mlflow.log_metric("cv_test_recall_mean", cv_scores["test_recall"].mean())

        if test_roc_auc is not None:
            mlflow.log_metric("test_roc_auc", float(test_roc_auc))

        if test_report is not None:
            mlflow.log_dict(test_report, "test_classification_report.json")

        with tempfile.NamedTemporaryFile("wb", delete=False) as temp_file:
            tmp_path = temp_file.name

        model_file_name = f"{ModelConfig.model_name}.pkl"
        with open(tmp_path, "wb") as f:
            pickle.dump(model, f)

        mlflow.log_artifact(tmp_path, artifact_path=RunConfig.run_name)
        os.remove(tmp_path)

        return run_id


def load_model_from_mlflow(run_id: str) -> BaseEstimator:
    mlflow.set_tracking_uri(RunConfig.mlflow_tracking_uri)

    experiment = mlflow.get_experiment_by_name(RunConfig.experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"Experiment {RunConfig.experiment_name} does not exist in MLflow."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=RunConfig.run_name,
            dst_path=tmp_dir,
        )

        model_file_name = f"{ModelConfig.model_name}.pkl"
        artifacts_path = os.path.join(tmp_dir, RunConfig.run_name)
        model_path = os.path.join(artifacts_path, model_file_name)

        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Model file {model_file_name} not found under MLflow artifacts."
            )

        with open(model_path, "rb") as f:
            model: BaseEstimator = pickle.load(f)

    return model