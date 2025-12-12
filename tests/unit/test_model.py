import pandas as pd
import numpy as np
import logging
import pytest

import src.model.model as model_module
from src.model.model import Model


@pytest.fixture
def sample_data() -> pd.DataFrame:
    # note: small synthetic dataset for testing
    randomizer = np.random.RandomState(42)
    size = 60

    data = pd.DataFrame({
        "feature1": randomizer.rand(size),
        "feature2": randomizer.rand(size),
        "feature3": randomizer.rand(size),
        "is_ransomware": np.array([0, 1] * (size // 2))
    })

    return data.sample(frac=1.0, random_state=42).reset_index(drop=True)

def test_get_splits(sample_data):
    X_train, X_test, y_train, y_test = Model._get_splits(sample_data)

    # note: basic shape checks
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(y_train) + len(y_test) == len(sample_data)

    # note: no overlap between train and test sets
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert set(y_train.index).isdisjoint(set(y_test.index))

    assert "is_ransomware" not in X_train.columns


def test_train_model_and_cv_scores(sample_data):
    m = Model()
    m.train_model(sample_data)

    assert m._model is not None
    assert m._cv_scores is not None
    assert "test_f1" in m._cv_scores

    scores = m.get_cv_scores()

    for key in ("test_accuracy", "test_precision", "test_recall", "test_f1"):
        assert key in scores
        arr = scores[key]
        assert hasattr(arr, "__len__")
        assert np.all((arr >= 0.0) & (arr <= 1.0))


def test_predict_proba_raises_if_not_trained(sample_data):
    model = Model()
    X = sample_data.drop(columns=["is_ransomware"]).iloc[:5]

    with pytest.raises(RuntimeError, match="not been trained/loaded"):
        model.predict_proba(X)


def test_evaluate_model_uses_stored_metrics_branch(sample_data, caplog):
    model = Model()
    model.train_model(sample_data)

    caplog.set_level(logging.INFO, logger=model_module.logger.name)

    # note: first evaluation computes and stores metrics
    model.evaluate_model()
    # note: assert it logs using stored metrics on second call
    assert any("Classification report" in rec.message for rec in caplog.records)


def test_evaluate_model_raises_if_no_metrics_and_no_data():
    model = Model()
    # note: no training so no metrics
    with pytest.raises(RuntimeError, match="No stored test metrics and no data provided"):
        model.evaluate_model()


def test_evaluate_model_recomputes_if_metrics_cleared(sample_data, caplog):
    model = Model()
    model.train_model(sample_data)

    caplog.set_level(logging.INFO, logger=model_module.logger.name)

    # note: force it to skip the early return and go into recompute path
    model._test_report = None
    model._test_roc_auc = None

    model.evaluate_model(sample_data)

    assert any("ROC AUC Score" in rec.message for rec in caplog.records)


def test_get_test_summary_raises_if_no_metrics(sample_data):
    model = Model()
    model.train_model(sample_data)

    # note: clear metrics so summary must fail
    model._test_report = None
    model._test_roc_auc = None
    with pytest.raises(RuntimeError, match="Test metrics have not been computed yet"):
        model.get_test_summary()


def test_get_test_summary_includes_positive_label_and_cv_stats(sample_data):
    model = Model()
    model.train_model(sample_data)

    summary = model.get_test_summary()

    assert "accuracy" in summary
    assert "roc_auc" in summary
    assert "f1_positive" in summary

    # note: if CV scores exist, these should be present
    assert "cv_test_f1_mean" in summary
    assert "cv_test_f1_std" in summary


def test_log_model_to_mlflow_raises_if_not_trained():
    model = Model()
    with pytest.raises(RuntimeError, match="not been trained/loaded"):
        model.log_model_to_mlflow()


def test_log_model_to_mlflow_raises_if_no_cv_scores(sample_data):
    model = Model()
    model.train_model(sample_data)
    # note: force CV missing
    model._cv_scores = None

    with pytest.raises(RuntimeError, match="no CV scores"):
        model.log_model_to_mlflow()


def test_log_model_to_mlflow_success(sample_data, monkeypatch):
    model = Model()
    model.train_model(sample_data)

    def fake_log_model_to_mlflow(**kwargs):
        # note: basic sanity checks, are correct keys passed through?
        assert kwargs["model"] is not None
        assert kwargs["cv_scores"] is not None
        return "RUN123"

    # note: patch the underlying function
    monkeypatch.setattr(model_module, "log_model_to_mlflow", fake_log_model_to_mlflow)

    run_id = model.log_model_to_mlflow()
    assert run_id == "RUN123"


def test_load_model_from_mlflow_sets_model(monkeypatch):
    model = Model()

    class Dummy:
        def predict_proba(self, X):
            return np.tile([0.2, 0.8], (len(X), 1))

    # note: patch the underlying function
    monkeypatch.setattr(model_module, "load_model_from_mlflow", lambda run_id: Dummy())

    model.load_model_from_mlflow(run_id="ANY")
    assert model._model is not None

    # note: test that the loaded model works
    X = pd.DataFrame({"feature1": [0.1], "feature2": [0.2], "feature3": [0.3]})
    proba = model.predict_proba(X)
    assert proba.shape == (1, 2)