import os
import pickle
import tempfile
from typing import Optional

import mlflow
from sklearn.base import BaseEstimator

from configs.configs import ModelConfig, RunConfig


def log_model_to_mlflow(model: BaseEstimator, cv_scores: dict, test_report: Optional[dict], test_roc_auc: Optional[float], test_accuracy: Optional[float]) -> str:
    mlflow.set_tracking_uri(RunConfig.mlflow_tracking_uri)
    mlflow.set_experiment(RunConfig.experiment_name)

    with mlflow.start_run(run_name=RunConfig.run_name) as run:
        run_id = run.info.run_id

        # note: cross-validation scores
        mlflow.log_dict(cv_scores, "cv_scores.json")
        if "test_f1" in cv_scores:
            mlflow.log_metric("cv_test_f1_mean", cv_scores["test_f1"].mean())
        if "test_precision" in cv_scores:
            mlflow.log_metric("cv_test_precision_mean", cv_scores["test_precision"].mean())
        if "test_recall" in cv_scores:
            mlflow.log_metric("cv_test_recall_mean", cv_scores["test_recall"].mean())

        # note: test metrics
        if test_roc_auc is not None:
            mlflow.log_metric("test_roc_auc", float(test_roc_auc))
        if test_accuracy is not None:
            mlflow.log_metric("test_accuracy", float(test_accuracy))
        if test_report is not None:
            mlflow.log_dict(test_report, "test_classification_report.json")

        model_file_name = f"{ModelConfig.model_name}.pkl"

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, model_file_name)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path, artifact_path=RunConfig.run_name)

        return run_id


def load_model_from_mlflow(run_id: str) -> BaseEstimator:
    mlflow.set_tracking_uri(RunConfig.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(RunConfig.experiment_name)

    if experiment is None:
        raise RuntimeError(f"Experiment {RunConfig.experiment_name} does not exist in MLflow.")

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
            raise RuntimeError(f"Model file {model_file_name} not found under MLflow artifacts.")

        with open(model_path, "rb") as f:
            model: BaseEstimator = pickle.load(f)

    return model


def get_latest_run_id(run_name: str) -> str:
    try:
        mlflow.set_tracking_uri(RunConfig.mlflow_tracking_uri)
        mlflow_experiment = mlflow.get_experiment_by_name(RunConfig.experiment_name)

        if mlflow_experiment is None:
            raise RuntimeError(f"Experiment {RunConfig.experiment_name} does not exist in MLflow.")

        runs = mlflow.search_runs(
            [mlflow_experiment.experiment_id],
            f"attributes.run_name = '{run_name}'"
        )

        if runs.empty:
            raise RuntimeError(f"Run with name {run_name} is not found in MLflow.")

        unfinished = runs[runs["end_time"].isna()]
        if len(unfinished) > 1:
            raise RuntimeError("MLflow has multiple unfinished runs.")

        if unfinished.empty:
            latest_run = runs.loc[runs["end_time"].idxmax()]
        else:
            latest_run = unfinished.iloc[0]

        return latest_run["run_id"]
    
    except Exception as e:
        print(f"[MLFLOW DEBUG] Failed to resolve latest run_id for '{run_name}': {e}")
        return None