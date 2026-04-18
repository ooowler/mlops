import json
from pathlib import Path

import pandas as pd

from fraud_detection.config import TARGET_COLUMN


def run_expectations(df: pd.DataFrame) -> dict:
    checks = {
        "column_class_exists": TARGET_COLUMN in df.columns,
        "class_values_binary": bool(df[TARGET_COLUMN].isin([0, 1]).all()) if TARGET_COLUMN in df.columns else False,
        "n_columns_31": len(df.columns) == 31,
        "row_count_positive": len(df) > 0,
        "fraud_rate_reasonable": (
            0 < float(df[TARGET_COLUMN].mean()) < 0.5 if TARGET_COLUMN in df.columns else False
        ),
    }
    checks["success"] = all(v for k, v in checks.items() if k != "success")
    return checks


def write_report(path: Path, df: pd.DataFrame) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    fraud_n = int((df[TARGET_COLUMN] == 1).sum()) if TARGET_COLUMN in df.columns else 0
    n = len(df)
    report = {
        "n_rows": n,
        "n_columns": len(df.columns),
        "fraud_count": fraud_n,
        "fraud_ratio": fraud_n / n if n else 0.0,
        "expectations": run_expectations(df),
    }
    try:
        import great_expectations as ge

        gds = ge.dataset.PandasDataset(df)
        gds.expect_column_values_to_be_in_set(TARGET_COLUMN, [0, 1])
        gds.expect_table_row_count_to_be_between(min_value=1)
        vr = gds.validate()
        report["expectations"]["great_expectations_success"] = bool(vr.success)
    except Exception as exc:
        report["expectations"]["great_expectations"] = {"skipped_or_error": str(exc)}
    path.write_text(json.dumps(report, indent=2))
    return report
