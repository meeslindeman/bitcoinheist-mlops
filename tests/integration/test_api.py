import numpy as np
import pytest

import src.main_api as main_api
from src.main_api import app


@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


# note: test the /health endpoint
def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


# note: test the /predict endpoint when no model is loaded
def test_predict_no_model(client, monkeypatch):
    def fake_get_model():
        raise RuntimeError("No trained model available.")

    monkeypatch.setattr(main_api, "get_model", fake_get_model, raising=False)

    resp = client.post("/predict", json={"f1": 1.0, "f2": 2.0})
    assert resp.status_code == 503
    data = resp.get_json()
    assert data["error"] == "No trained model available yet."


# note: test the /predict endpoint with invalid (non-JSON) input
def test_predict_invalid_input(client, monkeypatch):
    class FakeModel:
        def predict_proba(self, X):
            return np.array([[0.7, 0.3]])

    monkeypatch.setattr(main_api, "get_model", lambda: FakeModel(), raising=False)
    monkeypatch.setattr(main_api, "FEATURE_COLUMNS", ["f1", "f2"], raising=False)

    # note: text/plain so request.get_json() returns None
    # note: returns 400 error
    resp = client.post("/predict", data="not-json", content_type="text/plain")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "Invalid or missing JSON body"


# note: test the /predict endpoint with valid input and a mocked model
def test_predict_success(client, monkeypatch):
    class FakeModel:
        def predict_proba(self, X):
            return np.array([[0.6, 0.4]])

    class DummySparkDF:
        def __init__(self, pdf):
            self._pdf = pdf

        def toPandas(self):
            return self._pdf

    class DummySparkSession:
        def createDataFrame(self, pdf):
            return DummySparkDF(pdf)

    def fake_get_features(input_spark_df):
        return input_spark_df

    monkeypatch.setattr(main_api, "get_model", lambda: FakeModel(), raising=False)
    monkeypatch.setattr(main_api, "FEATURE_COLUMNS", ["f1", "f2"], raising=False)
    monkeypatch.setattr(main_api, "get_spark", lambda: DummySparkSession(), raising=False)
    monkeypatch.setattr(main_api, "get_features", fake_get_features, raising=False)

    resp = client.post("/predict", json={"f1": 1.0, "f2": 2.0})
    assert resp.status_code == 200
    data = resp.get_json()

    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] == "clean"
    assert abs(data["probability"] - 0.4) < 1e-6


# note: test the /predict endpoint when input JSON is missing some features
# note: should still work by filling in missing features with 0.0
def test_predict_success_with_missing_features(client, monkeypatch):
    class FakeModel:
        def predict_proba(self, X):
            return np.array([[0.2, 0.8]])

    class DummySparkDF:
        def __init__(self, pdf):
            self._pdf = pdf

        def toPandas(self):
            return self._pdf

    class DummySparkSession:
        def createDataFrame(self, pdf):
            return DummySparkDF(pdf)

    def fake_get_features(input_spark_df):
        return input_spark_df

    monkeypatch.setattr(main_api, "get_model", lambda: FakeModel(), raising=False)
    # note:pretend model expects an extra feature f3
    monkeypatch.setattr(main_api, "FEATURE_COLUMNS", ["f1", "f2", "f3"], raising=False)
    monkeypatch.setattr(main_api, "get_spark", lambda: DummySparkSession(), raising=False)
    monkeypatch.setattr(main_api, "get_features", fake_get_features, raising=False)

    # note: send JSON missing f3
    # note: API should add it as 0.0 and still work
    resp = client.post("/predict", json={"f1": 1.0, "f2": 2.0})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "prediction" in data
    assert "probability" in data


# note: test the /reload endpoint with mocking 
def test_reload_model_success(client, monkeypatch):
    class DummyModel:
        pass

    def fake_get_model():
        fake_get_model.called += 1
        return DummyModel()

    fake_get_model.called = 0
    # note: reload_model calls get_model.cache_clear()
    fake_get_model.cache_clear = lambda: None

    monkeypatch.setattr(main_api, "get_model", fake_get_model, raising=False)
    monkeypatch.setattr(
        main_api, "init_training_metrics_from_file", lambda: None, raising=False
    )

    resp = client.post("/reload")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert "Model reloaded" in data["message"]
    assert fake_get_model.called == 1


# note: if get_model fails during reload, /reload should return 500 with status 'error'
def test_reload_model_failure(client, monkeypatch):
    def failing_get_model():
        failing_get_model.cache_clear_called = True
        raise RuntimeError("Failed to reload model.")

    failing_get_model.cache_clear_called = False
    failing_get_model.cache_clear = lambda: None

    monkeypatch.setattr(main_api, "get_model", failing_get_model, raising=False)

    resp = client.post("/reload")
    assert resp.status_code == 500
    data = resp.get_json()
    assert data["status"] == "error"
    assert "Failed to reload model." in data["message"]