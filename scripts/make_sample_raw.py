import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_detection.config import DATA_RAW

COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    rng = np.random.default_rng(42)
    t = np.arange(n, dtype=np.float64)
    v = rng.standard_normal((n, 28))
    amt = np.abs(rng.standard_normal(n)) * 80.0
    y = (rng.random(n) < 0.02).astype(np.int8)
    fraud = y.astype(bool)
    v[fraud, 20:28] += 1.85
    amt = np.where(fraud, amt + 12.0, amt)
    df = pd.DataFrame(np.column_stack([t, v, amt, y]), columns=COLS)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    path = DATA_RAW / "creditcard.csv"
    df.to_csv(path, index=False)
    print(path, "rows", len(df))


if __name__ == "__main__":
    main()
