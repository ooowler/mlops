import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loguru import logger

from fraud_detection.config import DATA_RAW, DATA_PROCESSED, RANDOM_STATE, TEST_SIZE, TARGET_COLUMN
from fraud_detection.preprocess import run_from_raw_path


def main():
    run_from_raw_path(DATA_RAW / "creditcard.csv", DATA_PROCESSED, TEST_SIZE, RANDOM_STATE, TARGET_COLUMN)
    logger.info("preprocess done -> {}", DATA_PROCESSED)


if __name__ == "__main__":
    main()
