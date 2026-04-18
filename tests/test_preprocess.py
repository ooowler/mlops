import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from fraud_detection.data_loader import RAW_COLUMNS
from fraud_detection.preprocess import (
    coerce_feature_frame,
    run_from_raw_path,
    stratified_split_indices,
    write_train_test_files,
)


def _frame(class_vals, amount_vals=None, v1_vals=None):
    n = len(class_vals)
    d = {c: np.zeros(n, dtype=np.float64) for c in RAW_COLUMNS if c not in ("Class", "Amount")}
    d["Time"] = np.arange(n, dtype=np.float64)
    d["Class"] = np.asarray(class_vals, dtype=np.int64)
    if amount_vals is not None:
        d["Amount"] = amount_vals
    else:
        d["Amount"] = np.linspace(0.1, 2.0, n)
    if v1_vals is not None:
        d["V1"] = v1_vals
    return pd.DataFrame(d, columns=RAW_COLUMNS)


@pytest.mark.parametrize(
    "builder,expect_stratify_error",
    [
        (lambda: _frame([]), False),
        (lambda: _frame([1] * 50), True),
        (lambda: _frame([0] * 20 + [1] * 20, None, np.full(40, np.nan)), False),
        (lambda: _frame([0] * 15 + [1] * 15, ["x"] * 30, None), False),
    ],
    ids=["empty", "single_class", "all_nan_column", "amount_strings"],
)
def test_coerce_feature_frame_param(builder, expect_stratify_error):
    df = builder()
    out = coerce_feature_frame(df, "Class")
    if not len(df):
        assert len(out) == 0
        return
    assert "float" in str(out["Amount"].dtype).lower() or out["Amount"].dtype == np.float32
    if expect_stratify_error:
        with pytest.raises(ValueError):
            stratified_split_indices(out, 0.2, 42, "Class")
        return
    tr, te = stratified_split_indices(out, 0.2, 42, "Class")
    assert len(tr) + len(te) == len(out)


def test_stratified_split_indices_balanced():
    df = _frame([0] * 150 + [1] * 50)
    tr, te = stratified_split_indices(df, 0.25, 0, "Class")
    assert len(tr) + len(te) == 200
    assert set(tr) & set(te) == set()


def test_stratified_split_empty_raises():
    df = pd.DataFrame({c: [] for c in RAW_COLUMNS})
    with pytest.raises(ValueError):
        stratified_split_indices(df, 0.2, 0, "Class")


def test_write_train_test_files(tmp_path):
    df = _frame([0] * 80 + [1] * 20)
    tr, te = train_test_split(df.index, test_size=0.2, random_state=1, stratify=df["Class"])
    write_train_test_files(df, tr, te, tmp_path)
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "test.csv").exists()


def test_run_from_raw_path_roundtrip(tmp_path):
    raw = tmp_path / "creditcard.csv"
    out = tmp_path / "processed"
    df = _frame([0] * 100 + [1] * 20)
    df.to_csv(raw, index=False)
    run_from_raw_path(raw, out, 0.2, 7, "Class")
    assert (out / "train.csv").exists() and (out / "test.csv").exists()
    rec = pd.read_csv(out / "train.csv")
    assert len(rec) > 0 and "Class" in rec.columns
