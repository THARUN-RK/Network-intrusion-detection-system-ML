"""
Tests verifying leakage-free preprocessing, clean splits, and evaluation integrity.
"""

import numpy as np
import pandas as pd

import config
import preprocessing


def test_clean_features_drops_inf_nan_and_duplicates():
    X = pd.DataFrame({
        "f1": [1.0, 1.0, 2.0, np.inf, 3.0],
        "f2": [10.0, 10.0, 20.0, 30.0, np.nan],
    })
    y = pd.Series(["BENIGN", "BENIGN", "DoS", "DDoS", "Bot"])

    X_clean, y_clean, audit = preprocessing.clean_features_and_labels(
        X, y, drop_inf_nan=True, drop_duplicates=True
    )

    assert audit["raw_rows"] == 5
    assert audit["inf_nan_rows_dropped"] == 2
    assert audit["feature_duplicates_dropped"] == 1
    assert audit["clean_rows"] == 2
    assert len(X_clean) == 2
    assert len(y_clean) == 2


def test_stratified_clean_split_zero_overlap_on_benchmark():
    csv_path = config.DATASET_RAW_DIR / "CICIDS2017_Benchmark_Sample.csv"
    if not csv_path.exists():
        return

    raw_df = pd.read_csv(csv_path)
    X_raw, y_raw = preprocessing.preprocess_dataframe(raw_df)
    X_clean, y_clean, _ = preprocessing.clean_features_and_labels(X_raw, y_raw)
    X_tr, X_te, y_tr, y_te, checks = preprocessing.stratified_clean_split(
        X_clean, y_clean, test_size=0.30, random_state=config.RANDOM_STATE
    )

    assert checks["index_overlap"] == 0
    assert checks["feature_hash_overlap"] == 0

    X_bal, y_bal = preprocessing.build_balanced_test_set(X_te, y_te, random_state=config.RANDOM_STATE)
    counts = y_bal.value_counts()
    assert counts.nunique() == 1
    assert set(X_bal.index).issubset(set(X_te.index))
    assert len(set(X_bal.index) & set(X_tr.index)) == 0

    X_nat, y_nat, nat_meta = preprocessing.build_natural_test_set(
        X_te, y_te, target_benign_ratio=0.80, random_state=config.RANDOM_STATE
    )
    assert 0.78 <= nat_meta["benign_ratio"] <= 0.82
    assert set(X_nat.index).issubset(set(X_te.index))
    assert len(set(X_nat.index) & set(X_tr.index)) == 0
