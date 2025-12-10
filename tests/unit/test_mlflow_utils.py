import os
import pickle

import numpy as np
import pandas as pd
import pytest

import src.mlflow_utils as mlflow_utils
from configs.configs import ModelConfig, RunConfig


def test_log_model(monkeypatch):
    called = {
        "set_tracking_uri": False,
        "set_experiment": False,
        "start_run": False,
        "log_dict": [],
        "log_metric": [],
        "log_artifact": []
    }  

    # note: we have to fake a bunch of mlflow functions
    def fake_set_tracking_uri(uri):
        called["set_tracking_uri"] = True
        assert uri == RunConfig.mlflow_tracking_uri

    def fake_set_experiment(name):
        called["set_experiment"] = True
        assert name == RunConfig.experiment_name

    class FakeRun:
        def __init__(self):
            self.info = type("Info", (), {"run_id": "dummy-run-id"})

        def __enter__(self):
            called["start_run"] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            return False
    
    def fake_start_run(run_name):
        assert run_name == RunConfig.run_name
        return FakeRun()

    def fake_log_dict(obj, artifact_file):
        called["log_dict"].append(artifact_file)

    def fake_log_metric(name, value):
        called["log_metric"].append(name)

    def fake_log_artifact(path, artifact_path):
        called["log_artifact"].append((path, artifact_path))
    
    # note: replace mlflow functions with fakes
    monkeypatch.setattr(mlflow_utils.mlflow, "set_tracking_uri", fake_set_tracking_uri)
    monkeypatch.setattr(mlflow_utils.mlflow, "set_experiment", fake_set_experiment)
    monkeypatch.setattr(mlflow_utils.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(mlflow_utils.mlflow, "log_dict", fake_log_dict)
    monkeypatch.setattr(mlflow_utils.mlflow, "log_metric", fake_log_metric)
    monkeypatch.setattr(mlflow_utils.mlflow, "log_artifact", fake_log_artifact)

    cv_scores = {
        "test_f1": np.array([0.1, 0.2]),
    }
    fake_model = {"foo": "bar"}

    run_id = mlflow_utils.log_model_to_mlflow(
        model=fake_model,
        cv_scores=cv_scores,
        test_report={"dummy": 1},
        test_roc_auc=0.9,
        test_accuracy=0.8,
    )

    assert run_id == "dummy-run-id"
    assert called["set_tracking_uri"]
    assert called["set_experiment"]
    assert called["start_run"]
    assert "cv_scores.json" in called["log_dict"]
    assert "test_classification_report.json" in called["log_dict"]
    assert len(called["log_artifact"]) == 1
    logged_path, artifact_path = called["log_artifact"][0]
    assert artifact_path == RunConfig.run_name
    assert logged_path.endswith(f"{ModelConfig.model_name}.pkl")

def test_load_model(monkeypatch):
    dummy_model = {"loaded": True}

    class FakeExperiment:
        experiment_id = "exp123"

    def fake_set_tracking_uri(uri):
        pass

    def fake_get_experiment_by_name(name):
        return FakeExperiment()

    def fake_download_artifacts(run_id, artifact_path, dst_path):
        artifacts_dir = os.path.join(dst_path, RunConfig.run_name)
        os.makedirs(artifacts_dir, exist_ok=True)
        model_path = os.path.join(artifacts_dir, f"{ModelConfig.model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(dummy_model, f)

    monkeypatch.setattr(mlflow_utils.mlflow, "set_tracking_uri", fake_set_tracking_uri)
    monkeypatch.setattr(mlflow_utils.mlflow, "get_experiment_by_name", fake_get_experiment_by_name)
    monkeypatch.setattr(mlflow_utils.mlflow.artifacts, "download_artifacts", fake_download_artifacts)

    model = mlflow_utils.load_model_from_mlflow(run_id="run-123")
    assert model == dummy_model

def test_load_model_no_experiment(monkeypatch):
    def fake_set_tracking_uri(uri):
        pass

    def fake_get_experiment_by_name(name):
        return None

    monkeypatch.setattr(mlflow_utils.mlflow, "set_tracking_uri", fake_set_tracking_uri)
    monkeypatch.setattr(mlflow_utils.mlflow, "get_experiment_by_name", fake_get_experiment_by_name)

    with pytest.raises(RuntimeError):
        mlflow_utils.load_model_from_mlflow(run_id="any")

        
    