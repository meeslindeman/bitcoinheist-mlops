import pytest

from src.main_api import app

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


# note: check that /metrics is available and contains custom metrics
def test_metrics_endpoint_exposes_custom_metrics(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)

    # note: api custom metrics
    assert "prediction_latency_seconds" in body
    assert "prediction_requests_total" in body
    assert "prediction_errors_total" in body
    assert "prediction_labels" in body

    # note: training custom metrics
    assert "training_last_run_timestamp_seconds" in body
    assert "training_last_accuracy" in body
    assert "training_last_roc_auc" in body
    assert "training_last_f1_positive" in body