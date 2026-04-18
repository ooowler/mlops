import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, field_validator
import mlflow
import mlflow.sklearn

from fraud_detection.config import MODEL_DIR, FRAUD_CLASS
from fraud_detection.errors import ModelNotLoadedError, ModelType

EXPECTED_FEATURES = 30

app = FastAPI()


def disk_model_path():
    want = os.getenv("SERVE_MODEL", "").strip().lower()
    logp = MODEL_DIR / ModelType.LOGREG.value
    rfp = MODEL_DIR / ModelType.RF.value
    if want == ModelType.RF.value and rfp.exists():
        return rfp
    if want == ModelType.LOGREG.value and logp.exists():
        return logp
    if logp.exists():
        return logp
    return rfp if rfp.exists() else None


def load_serving_model():
    if uri := os.getenv("MLFLOW_MODEL_URI"):
        if turi := os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(turi)
        return mlflow.sklearn.load_model(uri)
    path = disk_model_path()
    return mlflow.sklearn.load_model(str(path)) if path else None


model = load_serving_model()


class Transaction(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"features": [0.0] * 29 + [100.0]}]},
    )
    features: list[float]
    Amount: float | None = None

    @field_validator("features")
    @classmethod
    def check_len(cls, v):
        if len(v) != EXPECTED_FEATURES:
            raise ValueError(f"expected {EXPECTED_FEATURES} features, got {len(v)}")
        return v


@app.get("/health")
def health():
    return {"status": "ok"}


@app.exception_handler(ModelNotLoadedError)
def handle_model_not_loaded(request, exc):
    return JSONResponse(status_code=503, content={"error": exc.message, "detail": exc.detail})


@app.post("/predict")
def predict(t: Transaction):
    if model is None:
        raise ModelNotLoadedError("model not loaded")
    row = [t.features]
    pred = model.predict(row)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0]
        order = list(model.classes_)
        idx = order.index(FRAUD_CLASS) if FRAUD_CLASS in order else 1
        fraud_probability = float(proba[idx])
    else:
        fraud_probability = float(int(pred) == FRAUD_CLASS)
    return {"fraud_probability": fraud_probability, "prediction": int(pred)}
