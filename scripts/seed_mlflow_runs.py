import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
import pandas as pd

from fraud_detection.config import DATA_PROCESSED, RANDOM_STATE, FRAUD_CLASS, TARGET_COLUMN
from fraud_detection.errors import ModelType

uri = os.getenv("MLFLOW_TRACKING_URI") or f"file://{ROOT / 'mlflow_data'}"
mlflow.set_tracking_uri(uri)

def run_one(run_name: str, model_type: str, model, X_train, y_train, X_test, y_test):
    mlflow.set_experiment("fraud_detection")
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        recall_fraud = recall_score(y_test, pred, pos_label=FRAUD_CLASS, zero_division=0)
        f1 = f1_score(y_test, pred, pos_label=FRAUD_CLASS, zero_division=0)
        mlflow.log_params({"model_type": model_type, "random_state": RANDOM_STATE, "class_weight": "balanced"})
        mlflow.log_metrics({"recall_fraud": recall_fraud, "f1_fraud": f1})

def main():
    train_path = DATA_PROCESSED / "train.csv"
    test_path = DATA_PROCESSED / "test.csv"
    train = pd.read_csv(train_path, nrows=3000)
    test = pd.read_csv(test_path, nrows=800)
    X_train = train.drop(columns=[TARGET_COLUMN])
    y_train = train[TARGET_COLUMN]
    X_test = test.drop(columns=[TARGET_COLUMN])
    y_test = test[TARGET_COLUMN]
    run_one(
        "logreg_balanced_seed3000",
        ModelType.LOGREG.value,
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"),
        X_train,
        y_train,
        X_test,
        y_test,
    )
    run_one(
        "rf_trees15_balanced_seed3000",
        ModelType.RF.value,
        RandomForestClassifier(n_estimators=15, random_state=RANDOM_STATE, class_weight="balanced"),
        X_train,
        y_train,
        X_test,
        y_test,
    )
    run_one(
        "rf_trees30_balanced_seed3000",
        ModelType.RF.value,
        RandomForestClassifier(n_estimators=30, random_state=RANDOM_STATE, class_weight="balanced"),
        X_train,
        y_train,
        X_test,
        y_test,
    )

if __name__ == "__main__":
    main()
