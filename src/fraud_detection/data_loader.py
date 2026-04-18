from pathlib import Path
import pandas as pd

from fraud_detection.config import TARGET_COLUMN

RAW_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def load_raw(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8", errors="replace") as f:
        first = f.readline().lstrip("\ufeff")
    if first.strip().upper().startswith("@"):
        skip = None
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if line.strip().upper() == "@DATA":
                    skip = i + 1
                    break
        if skip is None:
            raise ValueError("No @DATA in ARFF file")
        df = pd.read_csv(path, skiprows=skip, header=None, names=RAW_COLUMNS, encoding="utf-8")
    else:
        df = pd.read_csv(path, encoding="utf-8")
        if list(df.columns) != RAW_COLUMNS:
            if not set(RAW_COLUMNS).issubset(df.columns):
                raise ValueError(f"Unexpected columns: {list(df.columns)}")
            df = df[list(RAW_COLUMNS)]
    feat_cols = [c for c in df.columns if c != TARGET_COLUMN]
    df[feat_cols] = df[feat_cols].astype("float32")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int8")
    return df
