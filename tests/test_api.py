from fraud_detection.api import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_no_model():
    r = client.post("/predict", json={"features": [0.0] * 30})
    out = r.json()
    assert r.status_code == 503 or "prediction" in out
    if r.status_code == 503:
        assert "error" in out
