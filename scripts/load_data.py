import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from loguru import logger

from fraud_detection.config import DATA_RAW
from fraud_detection.data_loader import load_raw
from fraud_detection.data_quality import write_report
from fraud_detection.errors import DataKind, DataNotFoundError

REPORT_PATH = ROOT / "data" / "meta" / "raw_report.json"


def main():
    raw_path = DATA_RAW / "creditcard.csv"
    if not raw_path.exists():
        raise DataNotFoundError(f"Put creditcard.csv in {DATA_RAW}", detail=str(DataKind.RAW.value))
    df = load_raw(raw_path)
    report = write_report(REPORT_PATH, df)
    if not report["expectations"]["success"]:
        raise ValueError(f"data quality failed: {report['expectations']}")
    logger.info(
        "rows={} fraud_ratio={:.6f} report={}",
        report["n_rows"],
        report["fraud_ratio"],
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()
