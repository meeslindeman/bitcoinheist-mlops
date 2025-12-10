import numpy as np
from sklearn.ensemble import RandomForestClassifier

from configs.configs import RunConfig
from src.mlflow_utils import log_model_to_mlflow, load_model_from_mlflow, get_latest_run_id


# note: helper to create a tiny trained model for testing
def _tiny_model():
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    clf = RandomForestClassifier(n_estimators=1, max_depth=2, random_state=0)
    clf.fit(X, y)
    return clf, X


# note: test logging a model to MLflow and loading it back
def test_log_and_load_model(monkeypatch):
    original_experiment = RunConfig.experiment_name
    original_run_name = RunConfig.run_name

    test_experiment = f"{original_experiment}_integration"
    test_run_name = "integration_log_load"

    # note: we need to patch this to not interfere with productions runs
    monkeypatch.setattr(RunConfig, "experiment_name", test_experiment, raising=False)
    monkeypatch.setattr(RunConfig, "run_name", test_run_name, raising=False)

    model, X = _tiny_model()

    cv_scores = {"test_f1": np.array([0.5, 0.6])}
    test_report = {"1": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 1}}

    run_id = log_model_to_mlflow(
        model=model,
        cv_scores=cv_scores,
        test_report=test_report,
        test_roc_auc=0.9,
        test_accuracy=0.8,
    )

    assert isinstance(run_id, str)
    assert run_id != ""

    loaded = load_model_from_mlflow(run_id=run_id)

    # note: loaded model behaves like the original on the tiny dataset
    preds_original = model.predict(X)
    preds_loaded = loaded.predict(X)
    assert np.array_equal(preds_original, preds_loaded)


# note: test getting the latest run ID from MLflow
def test_get_latest_run_id(monkeypatch):
    original_experiment = RunConfig.experiment_name
    original_run_name = RunConfig.run_name

    test_experiment = f"{original_experiment}_integration"
    test_run_name = "integration_log_load"

    # note: we need to patch this to not interfere with productions runs
    monkeypatch.setattr(RunConfig, "experiment_name", test_experiment, raising=False)
    monkeypatch.setattr(RunConfig, "run_name", test_run_name, raising=False)

    model, _ = _tiny_model()

    cv_scores = {"test_f1": np.array([0.3, 0.4])}
    test_report = {
        "0": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 1}
    }

    new_run_id = log_model_to_mlflow(
        model=model,
        cv_scores=cv_scores,
        test_report=test_report,
        test_roc_auc=0.85,
        test_accuracy=0.75,
    )

    latest = get_latest_run_id(test_run_name)
    assert latest == new_run_id