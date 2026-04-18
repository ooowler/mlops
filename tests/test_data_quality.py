import json

import pandas as pd

from fraud_detection.config import TARGET_COLUMN
from fraud_detection.data_quality import run_expectations, write_report


def test_run_expectations_ok():
    df = pd.DataFrame({TARGET_COLUMN: [0] * 40 + [1] * 5})
    df["Time"] = 0.0
    for i in range(1, 29):
        df[f"V{i}"] = 0.0
    df["Amount"] = 1.0
    assert len(df.columns) == 31
    rep = run_expectations(df)
    assert rep["column_class_exists"] is True


def test_write_report_json(tmp_path):
    df = pd.DataFrame({TARGET_COLUMN: [0] * 50 + [1] * 10})
    df["Time"] = 0.0
    for i in range(1, 29):
        df[f"V{i}"] = 0.0
    df["Amount"] = 1.0
    p = tmp_path / "r.json"
    out = write_report(p, df)
    assert p.exists()
    assert out["n_rows"] == 60
    loaded = json.loads(p.read_text())
    assert loaded["n_rows"] == 60
