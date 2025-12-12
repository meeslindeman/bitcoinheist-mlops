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

        # note: if CV score arrays/series are present, log their means as summary metrics
        # note: reminder that these are validation scores from cross-validation, not test scores but I want to keep naming consistent
        if "test_f1" in cv_scores:
            mlflow.log_metric("cv_test_f1_mean", cv_scores["test_f1"].mean())
        if "test_precision" in cv_scores:
            mlflow.log_metric("cv_test_precision_mean", cv_scores["test_precision"].mean())
        if "test_recall" in cv_scores:
            mlflow.log_metric("cv_test_recall_mean", cv_scores["test_recall"].mean())

        # note: log test metrics
        if test_roc_auc is not None:
            mlflow.log_metric("test_roc_auc", float(test_roc_auc))
        if test_accuracy is not None:
            mlflow.log_metric("test_accuracy", float(test_accuracy))
        if test_report is not None:
            mlflow.log_dict(test_report, "test_classification_report.json")

        # note: persist the model to a temporary file and upload as an artifact
        model_file_name = f"{ModelConfig.model_name}.pkl"

        # note: trained model lives in memory so we need to serialize it to a file first before logging to MLflow
        # note: use a temporary directory to ensure each run gets a clean file and no leftovers remain on disk
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, model_file_name)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path, artifact_path=RunConfig.run_name)

        return run_id


def load_model_from_mlflow(run_id: str) -> BaseEstimator:
    mlflow.set_tracking_uri(RunConfig.mlflow_tracking_uri)

    # note: note strictly necessary but catch misconfiguration early
    experiment = mlflow.get_experiment_by_name(RunConfig.experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment {RunConfig.experiment_name} does not exist in MLflow.")

    # note: download artifacts for the given run into a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=RunConfig.run_name,
            dst_path=tmp_dir,
        )

        model_file_name = f"{ModelConfig.model_name}.pkl"
        artifacts_path = os.path.join(tmp_dir, RunConfig.run_name)
        model_path = os.path.join(artifacts_path, model_file_name)

        # note: ensure the model file exists
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model file {model_file_name} not found under MLflow artifacts.")

        # note: load and return the scikit-learn estimator (kept ambiguous to allow other models in future)
        with open(model_path, "rb") as f:
            model: BaseEstimator = pickle.load(f)

    return model


def get_latest_run_id(run_name: str) -> str:
    mlflow.set_tracking_uri(RunConfig.mlflow_tracking_uri)
    mlflow_experiment = mlflow.get_experiment_by_name(RunConfig.experiment_name)
    
    if mlflow_experiment is None:
        raise RuntimeError(f"Experiment {RunConfig.experiment_name} does not exist in MLflow.")

    # note: query MLflow for runs with the requested run name
    runs = mlflow.search_runs(
        [mlflow_experiment.experiment_id],
        f"attributes.run_name = '{run_name}'"
    )

    if runs.empty:
        raise RuntimeError(f"Run with name {run_name} is not found in MLflow.")

    unfinished = runs[runs["end_time"].isna()]
    if len(unfinished) > 1:
        # note: too many unfinished runs, this should not happen in normal circumstances so something is wrong
        raise RuntimeError("MLflow has multiple unfinished runs. Expected one or none unfinished run.")
    if unfinished.empty:
        # note: if no unfinished runs, return the latest finished run
        latest_run = runs.loc[runs["end_time"].idxmax()]
    else:
        # note: if one unfinished run, return that one
        latest_run = unfinished.iloc[0]

    return latest_run["run_id"]