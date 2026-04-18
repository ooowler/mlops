import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = Path(os.getenv("DATA_RAW", str(PROJECT_ROOT / "data" / "raw")))
DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED", str(PROJECT_ROOT / "data" / "processed")))
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(PROJECT_ROOT / "models")))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
FRAUD_CLASS = int(os.getenv("FRAUD_CLASS", "1"))
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "Class")
