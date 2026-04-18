import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from loguru import logger

import fraud_detection.config as CFG
from fraud_detection.config import DATA_PROCESSED, MODEL_DIR, RANDOM_STATE, FRAUD_CLASS, TARGET_COLUMN
from fraud_detection.errors import DataNotFoundError, ModelType

if uri := os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(uri)
else:
    mlflow.set_tracking_uri((ROOT / "mlflow_data").resolve().as_uri())


def _models(n_estimators, random_state, logreg_max_iter):
    return {
        ModelType.RF.value: RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        ),
        ModelType.LOGREG.value: LogisticRegression(
            max_iter=logreg_max_iter,
            random_state=random_state,
            class_weight="balanced",
        ),
    }


def _int_or_default(s):
    v = int(s)
    return None if v < 0 else v


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=[m.value for m in ModelType], default=ModelType.RF.value)
    p.add_argument("--experiment", default="fraud_detection")
    p.add_argument("--max-rows", type=_int_or_default, default=None, help="max rows; -1 = all")
    p.add_argument("--n-estimators", type=_int_or_default, default=None, help="-1 = auto from max_rows")
    p.add_argument("--random-state", type=_int_or_default, default=None, help="-1 = config RANDOM_STATE")
    p.add_argument("--logreg-max-iter", type=int, default=1000)
    return p.parse_args()


def run_name_for(model_type: str, n_estimators: int, max_rows):
    tag = f"trainrows{max_rows}" if max_rows else "fulldata"
    if model_type == ModelType.RF.value:
        return f"rf_trees{n_estimators}_balanced_{tag}"
    return f"logreg_balanced_{tag}"


def load_data(max_rows=None):
    train_path = DATA_PROCESSED / "train.csv"
    test_path = DATA_PROCESSED / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise DataNotFoundError("Run preprocess.py first")
    nrows_train = max_rows if max_rows else None
    nrows_test = (max_rows // 4) if max_rows else None
    train = pd.read_csv(train_path, nrows=nrows_train)
    test = pd.read_csv(test_path, nrows=nrows_test)
    X_train = train.drop(columns=[TARGET_COLUMN])
    y_train = train[TARGET_COLUMN]
    X_test = test.drop(columns=[TARGET_COLUMN])
    y_test = test[TARGET_COLUMN]
    return X_train, y_train, X_test, y_test


def train_and_log(args):
    X_train, y_train, X_test, y_test = load_data(args.max_rows)
    mlflow.set_experiment(args.experiment)
    rs = args.random_state if args.random_state is not None else RANDOM_STATE
    default_n = 10 if args.max_rows and args.max_rows < 20000 else 100
    n_est = args.n_estimators if args.n_estimators is not None else default_n
    models = _models(n_est, rs, args.logreg_max_iter)
    with mlflow.start_run(run_name=run_name_for(args.model, n_est, args.max_rows)):
        model = models[args.model]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        recall_fraud = recall_score(y_test, pred, pos_label=FRAUD_CLASS, zero_division=0)
        f1 = f1_score(y_test, pred, pos_label=FRAUD_CLASS, zero_division=0)
        params = {f"arg_{k}": ("none" if v is None else v) for k, v in vars(args).items()}
        params["resolved_n_estimators"] = n_est if args.model == ModelType.RF.value else 0
        params["resolved_random_state"] = rs
        params["config_RANDOM_STATE"] = CFG.RANDOM_STATE
        params["config_TEST_SIZE"] = CFG.TEST_SIZE
        params["config_FRAUD_CLASS"] = CFG.FRAUD_CLASS
        params["config_TARGET_COLUMN"] = CFG.TARGET_COLUMN
        params["class_weight"] = "balanced"
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({"recall_fraud": recall_fraud, "f1_fraud": f1})
        mlflow.sklearn.log_model(model, "model")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        out = MODEL_DIR / args.model
        if out.exists():
            shutil.rmtree(out)
        mlflow.sklearn.save_model(model, str(out))
    logger.info("recall_fraud={:.4f} f1_fraud={:.4f}", recall_fraud, f1)
    return recall_fraud, f1


def main():
    args = parse_args()
    train_and_log(args)


if __name__ == "__main__":
    main()
