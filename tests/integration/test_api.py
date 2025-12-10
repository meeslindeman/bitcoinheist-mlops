import pytest

from src.app.main_api import app


@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


# note: test the /health endpoint
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data == {"status": "ok"}


# note: test the /predict endpoint with invalid (non-JSON) input
def test_predict_invalid_input(client):
    resp = client.post("/predict", data="not-json", content_type="text/plain")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "Invalid or missing JSON body"

# note: test the /predict endpoint with valid input and a mocked model
def test_predict_success(client):
    input = {
        "year": 2014,
        "day": 150,
        "length": 5,
        "weight": 0.12,
        "count": 3,
        "looped": 1,
        "neighbors": 15,
        "income": 0.003,
    }
    resp = client.post("/predict", json=input)
    assert resp.status_code == 200

    data = resp.get_json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in {"ransomware", "clean"}
    assert 0.0 <= data["probability"] <= 1.0

# note: test the /predict endpoint when input JSON is missing some features
# note: should still work by filling in missing features with 0.0
def test_predict_with_missing_feature(client):
    input = {
        "year": 2014,
        "day": 150,
        "length": 5,
        "weight": 0.12,
        "count": 3,
        "looped": 1,
        "neighbors": 15,
        "income": 0.003,
    }
    input.pop("income")  

    resp = client.post("/predict", json=input)
    assert resp.status_code == 400


# note: if an unknown feature is provided, the API should ignore it
def test_predict_with_unknown_feature(client):
    input = {
        "year": 2014,
        "day": 150,
        "length": 5,
        "weight": 0.12,
        "count": 3,
        "looped": 1,
        "neighbors": 15,
        "income": 0.003,
    }
    input["unknown_feature"] = 123

    resp = client.post("/predict", json=input)
    assert resp.status_code == 200

    data = resp.get_json()
    assert "prediction" in data
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0


# note: test the /reload endpoint to reload the model
def test_reload_model_success(client):
    resp = client.post("/reload")
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["status"] == "ok"
    assert "Model reloaded" in data["message"]