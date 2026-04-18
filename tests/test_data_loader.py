import numpy as np
import pandas as pd

from fraud_detection.data_loader import RAW_COLUMNS, load_raw


def test_load_raw_csv_roundtrip(tmp_path):
    p = tmp_path / "c.csv"
    df = pd.DataFrame(np.zeros((12, 31)), columns=RAW_COLUMNS)
    df["Class"] = [0] * 10 + [1] * 2
    df["Amount"] = np.linspace(0.1, 3.0, 12)
    df.to_csv(p, index=False)
    got = load_raw(p)
    assert len(got) == 12
    assert list(got.columns) == RAW_COLUMNS
