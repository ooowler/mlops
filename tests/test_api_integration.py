import statistics
import time

from fastapi.testclient import TestClient

from fraud_detection.api import app


def test_predict_happy_path_fraud_probability_range(client_with_model):
    body = {"features": [0.1] * 29 + [50.0]}
    r = client_with_model.post("/predict", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "fraud_probability" in data
    p = data["fraud_probability"]
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_predict_missing_features_422():
    r = TestClient(app).post("/predict", json={})
    assert r.status_code == 422


def test_predict_amount_wrong_type_422():
    body = {"features": [0.0] * 30, "Amount": "abc"}
    r = TestClient(app).post("/predict", json=body)
    assert r.status_code == 422


def test_predict_wrong_feature_dim_422():
    r = TestClient(app).post("/predict", json={"features": [0.0] * 12})
    assert r.status_code == 422


def test_predict_wrong_element_type_in_features_422():
    r = TestClient(app).post("/predict", json={"features": [0.0] * 29 + ["nope"]})
    assert r.status_code == 422


def test_predict_load_median_latency(client_with_model):
    body = {"features": [0.05 * i for i in range(30)]}
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        r = client_with_model.post("/predict", json=body)
        times.append(time.perf_counter() - t0)
        assert r.status_code == 200
    assert statistics.median(times) < 0.2
