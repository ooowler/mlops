import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.data_loader import load_raw
from fraud_detection.errors import DataKind, DataNotFoundError


def coerce_feature_frame(df, target_column):
    if not len(df):
        return df.copy()
    out = df.copy()
    target_values = pd.to_numeric(out[target_column], errors="coerce")
    out[target_column] = target_values.fillna(0).astype("int8")
    for c in out.columns:
        if c == target_column:
            continue
        col = pd.to_numeric(out[c], errors="coerce")
        if col.isna().all():
            out[c] = 0.0
        else:
            med = col.median()
            fill = med if pd.notna(med) else 0.0
            out[c] = col.fillna(fill).astype("float32")
    return out


def stratified_split_indices(df, test_size, random_state, target_column):
    y = df[target_column]
    if y.nunique() < 2:
        raise ValueError("stratified split needs at least two classes")
    return train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def write_train_test_files(df, train_idx, test_idx, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    df.loc[train_idx].to_csv(out_dir / "train.csv", index=False)
    df.loc[test_idx].to_csv(out_dir / "test.csv", index=False)


def run_from_raw_path(raw_path, out_dir, test_size, random_state, target_column):
    if not raw_path.exists():
        raise DataNotFoundError(f"Put creditcard.csv in {raw_path.parent}", detail=str(DataKind.RAW.value))
    df = load_raw(raw_path)
    train_idx, test_idx = stratified_split_indices(df, test_size, random_state, target_column)
    write_train_test_files(df, train_idx, test_idx, out_dir)
