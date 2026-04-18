import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score

from fraud_detection.config import RANDOM_STATE, FRAUD_CLASS


def _fake_data(n=1000, fraud_ratio=0.01):
    n_fraud = int(n * fraud_ratio)
    X = np.random.randn(n, 5).astype(np.float32)
    y = np.zeros(n, dtype=int)
    y[-n_fraud:] = 1
    return X, y


def test_rf_fit_predict():
    X, y = _fake_data()
    model = RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    r = recall_score(y, pred, pos_label=FRAUD_CLASS, zero_division=0)
    f1 = f1_score(y, pred, pos_label=FRAUD_CLASS, zero_division=0)
    assert 0 <= r <= 1 and 0 <= f1 <= 1


def test_logreg_fit_predict():
    X, y = _fake_data()
    model = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
