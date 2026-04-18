import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from fraud_detection.config import RANDOM_STATE


@pytest.fixture
def fitted_logreg():
    X = [[float(i % 7) + j * 0.01 for j in range(30)] for i in range(40)]
    y = [0] * 35 + [1] * 5
    m = LogisticRegression(max_iter=800, random_state=RANDOM_STATE)
    m.fit(X, y)
    return m


@pytest.fixture
def client_with_model(fitted_logreg):
    import fraud_detection.api as api_module

    prev = api_module.model
    api_module.model = fitted_logreg
    yield TestClient(api_module.app)
    api_module.model = prev
