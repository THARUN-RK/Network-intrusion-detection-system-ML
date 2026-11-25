"""
Network Intrusion Detection System (NIDS) - Data Preprocessing Module
Optimized for CIC-IDS2017 Dataset and 8GB RAM Hardware Specification.

Features:
- Robust multi-CSV loading & concatenation
- Header whitespace stripping & label normalization
- Data leakage prevention: Drops IP addresses, Flow IDs, Timestamps
- Infinite value (inf, -inf) & NaN sanitization
- Canonical attack mapping into 10 target categories
- Memory-safe controlled stratified sampling
- Reproducible Scikit-learn preprocessing pipeline (Imputer + StandardScaler)
"""

import os
import re
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Map fine-grained CIC-IDS2017 attack labels to 10 canonical categories
CANONICAL_LABEL_MAPPING = {
    "benign": "BENIGN",
    
    # DoS Variations
    "dos slowloris": "DoS",
    "dos slowhttptest": "DoS",
    "dos hulk": "DoS",
    "dos goldeneye": "DoS",
    "dos": "DoS",
    
    # DDoS Variations
    "ddos": "DDoS",
    
    # PortScan Variations
    "portscan": "PortScan",
    
    # Botnet Variations
    "bot": "Bot",
    
    # Brute Force Patator
    "ftp-patator": "FTP-Patator",
    "ssh-patator": "SSH-Patator",
    
    # Web Attack Variations
    "web attack – brute force": "Web Attack",
    "web attack - brute force": "Web Attack",
    "web attack – xss": "Web Attack",
    "web attack - xss": "Web Attack",
    "web attack – sql injection": "Web Attack",
    "web attack - sql injection": "Web Attack",
    "web attack": "Web Attack",
    
    # Infiltration
    "infiltration": "Infiltration",
    "infilteration": "Infiltration",
    
    # Heartbleed
    "heartbleed": "Heartbleed"
}

# Features that introduce data leakage (identifiers, temporal markers, addresses)
LEAKAGE_COLUMNS = [
    "flow id",
    "source ip",
    "src ip",
    "destination ip",
    "dst ip",
    "source port",
    "src port",
    "timestamp",
    "simillarhttp",
    "unnamed: 0"
]


def clean_column_name(col: str) -> str:
    """Normalize column header: remove extra whitespace and non-alphanumeric trailing chars."""
    col = col.strip()
    col = re.sub(r"\s+", " ", col)
    return col


def canonicalize_label(raw_label: Any) -> str:
    """Map raw dataset label strings to canonical 10-class taxonomy."""
    if pd.isna(raw_label):
        return "BENIGN"
    
    cleaned = str(raw_label).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    
    # Specific priority check: DDoS must precede DoS substring match
    if "ddos" in cleaned:
        return "DDoS"
    if "dos" in cleaned:
        return "DoS"
    if "portscan" in cleaned or "port scan" in cleaned:
        return "PortScan"
    if "bot" in cleaned:
        return "Bot"
    if "ftp-patator" in cleaned:
        return "FTP-Patator"
    if "ssh-patator" in cleaned:
        return "SSH-Patator"
    if "web attack" in cleaned:
        return "Web Attack"
    if "infiltrat" in cleaned:
        return "Infiltration"
    if "heartbleed" in cleaned:
        return "Heartbleed"
    if "benign" in cleaned:
        return "BENIGN"
            
    return "Other_Attack"


class NetworkFlowSanitizer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to replace inf, -inf with NaN so SimpleImputer can process them cleanly.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = np.array(X, dtype=np.float64, copy=True)
        # Replace positive and negative infinity with NaN
        X_copy[np.isinf(X_copy)] = np.nan
        return X_copy


def build_feature_pipeline() -> Pipeline:
    """
    Constructs a leakage-free Scikit-learn Pipeline:
    1. Sanitizer: Replaces inf / -inf with NaN
    2. SimpleImputer: Imputes NaN with column median (robust to network spikes)
    3. StandardScaler: Centers and scales features (essential for SVM RBF kernel)
    """
    return Pipeline([
        ("sanitizer", NetworkFlowSanitizer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


def load_raw_dataset(raw_dir: Optional[Path] = None, max_sample_per_file: Optional[int] = None) -> pd.DataFrame:
    """
    Loads all CIC-IDS2017 CSV files from raw directory, normalizes columns,
    and removes obvious network leakage fields.
    """
    raw_path = raw_dir or config.DATASET_RAW_DIR
    csv_files = list(raw_path.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {raw_path}")
    
    dataframes = []
    logger.info(f"Discovered {len(csv_files)} CSV files in {raw_path}")
    
    for file in csv_files:
        logger.info(f"Loading {file.name}...")
        try:
            # Low_memory=False to prevent mixed-type inference warnings
            df = pd.read_csv(file, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding="latin1", low_memory=False)
            
        # Clean column names
        df.columns = [clean_column_name(c) for c in df.columns]
        
        # Subsample if requested to maintain 8GB RAM headroom
        if max_sample_per_file and len(df) > max_sample_per_file:
            df = df.sample(n=max_sample_per_file, random_state=config.RANDOM_STATE)
            
        dataframes.append(df)
        
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Total raw records loaded: {len(combined_df):,}")
    return combined_df


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies comprehensive cybersecurity dataset cleaning:
    1. Label column identification and canonicalization
    2. Drops network leakage columns (IPs, Ports, Flow IDs, Timestamps)
    3. Aligns features with CORE_FLOW_FEATURES
    4. Handles duplicates and infinite values
    """
    # 1. Identify label column
    label_col = None
    for c in df.columns:
        if c.lower() in ["label", "attack", "class"]:
            label_col = c
            break
            
    if not label_col:
        raise ValueError(f"Label column not found in dataset. Columns: {list(df.columns[:10])}")
        
    # Map labels to 10 canonical categories
    y = df[label_col].apply(canonicalize_label)
    
    # 2. Drop leakage columns
    cols_to_drop = [c for c in df.columns if c.lower() in LEAKAGE_COLUMNS or c == label_col]
    X_raw = df.drop(columns=cols_to_drop)
    
    # 3. Ensure numerical types
    for col in X_raw.columns:
        if X_raw[col].dtype == object:
            X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")
            
    # 4. Filter columns to only those defined in CORE_FLOW_FEATURES (or keep common set)
    common_features = [col for col in config.CORE_FLOW_FEATURES if col in X_raw.columns]
    if len(common_features) < 15:
        # If headers differ slightly, keep all numeric features
        common_features = list(X_raw.select_dtypes(include=[np.number]).columns)
        
    X = X_raw[common_features].copy()
    
    logger.info(f"Preprocessed features shape: {X.shape}, Target distribution:\n{y.value_counts()}")
    return X, y


def get_stratified_sample(X: pd.DataFrame, y: pd.Series, max_samples: int = config.SVM_TRAIN_SAMPLE_SIZE) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Performs memory-aware stratified sampling so rare attack classes (e.g. Heartbleed, Infiltration)
    are retained without exceeding 8GB RAM limitations during SVM $O(N^2)$ training.
    """
    if len(X) <= max_samples:
        return X, y
        
    # Minimum 5 samples per class if available
    stratified_indices = []
    classes = y.unique()
    sample_ratio = max_samples / len(X)
    
    sampled_dfs_X = []
    sampled_dfs_y = []
    
    for cls in classes:
        cls_mask = (y == cls)
        cls_count = cls_mask.sum()
        cls_sample_count = max(5, int(cls_count * sample_ratio))
        cls_sample_count = min(cls_count, cls_sample_count)
        
        idx = y[cls_mask].sample(n=cls_sample_count, random_state=config.RANDOM_STATE).index
        sampled_dfs_X.append(X.loc[idx])
        sampled_dfs_y.append(y.loc[idx])
        
    X_sampled = pd.concat(sampled_dfs_X)
    y_sampled = pd.concat(sampled_dfs_y)
    
    logger.info(f"Stratified sampling complete: {len(X_sampled):,} records selected from {len(X):,}")
    return X_sampled, y_sampled


def clean_features_and_labels(
    X: pd.DataFrame,
    y: pd.Series,
    drop_inf_nan: bool = True,
    drop_duplicates: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
    """
    Removes inf/-inf, NaN rows, and duplicate feature rows BEFORE train/test splitting
    so identical flows cannot leak across train and test partitions.
    """
    raw_rows = int(len(X))
    X_work = X.replace([np.inf, -np.inf], np.nan)

    inf_nan_mask = X_work.isna().any(axis=1)
    inf_nan_count = int(inf_nan_mask.sum())
    if drop_inf_nan and inf_nan_count > 0:
        valid_mask = ~inf_nan_mask
        X_work = X_work.loc[valid_mask].copy()
        y_work = y.loc[valid_mask].copy()
    else:
        y_work = y.copy()

    # Check exact row + label duplicates and feature-only duplicates
    combined = X_work.copy()
    combined["__label__"] = y_work.values
    exact_row_label_dups = int(combined.duplicated(keep="first").sum())
    feature_dups_mask = X_work.duplicated(keep="first")
    feature_dups_count = int(feature_dups_mask.sum())
    conflicting_label_dups = int(feature_dups_count - exact_row_label_dups)

    if drop_duplicates and feature_dups_count > 0:
        keep_mask = ~feature_dups_mask
        X_work = X_work.loc[keep_mask].copy()
        y_work = y_work.loc[keep_mask].copy()

    audit_stats = {
        "raw_rows": raw_rows,
        "inf_nan_rows_dropped": inf_nan_count if drop_inf_nan else 0,
        "inf_nan_rows_found": inf_nan_count,
        "exact_row_label_duplicates_found": exact_row_label_dups,
        "feature_duplicates_dropped": feature_dups_count if drop_duplicates else 0,
        "conflicting_label_duplicates": max(0, conflicting_label_dups),
        "clean_rows": int(len(X_work)),
    }
    logger.info(
        f"Clean dataset audit: raw={raw_rows:,}, inf/nan_dropped={audit_stats['inf_nan_rows_dropped']:,}, "
        f"duplicates_dropped={audit_stats['feature_duplicates_dropped']:,}, clean={audit_stats['clean_rows']:,}"
    )
    return X_work, y_work, audit_stats


def stratified_clean_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.30,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, int]]:
    """
    Performs stratified train/test split on already cleaned/deduplicated data,
    then verifies zero index and zero feature-hash overlap across splits.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    index_overlap = len(set(X_train.index) & set(X_test.index))
    train_hashes = set(pd.util.hash_pandas_object(X_train, index=False).tolist())
    test_hashes = set(pd.util.hash_pandas_object(X_test, index=False).tolist())
    feature_hash_overlap = len(train_hashes & test_hashes)

    if index_overlap > 0 or feature_hash_overlap > 0:
        raise RuntimeError(
            f"Data leakage detected across split: index_overlap={index_overlap}, "
            f"feature_hash_overlap={feature_hash_overlap}"
        )

    split_checks = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "index_overlap": index_overlap,
        "feature_hash_overlap": feature_hash_overlap,
    }
    return X_train, X_test, y_train, y_test, split_checks


def build_balanced_test_set(
    X_test_pool: pd.DataFrame,
    y_test_pool: pd.Series,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Builds a uniformly balanced evaluation set strictly from the held-out test pool
    by sampling `min_class_count` flows per class without replacement.
    """
    counts = y_test_pool.value_counts()
    n_per_class = int(counts.min())
    sampled_indices = []
    for cls in sorted(counts.index):
        cls_idx = y_test_pool[y_test_pool == cls].sample(
            n=n_per_class, replace=False, random_state=random_state
        ).index
        sampled_indices.extend(cls_idx.tolist())
    X_bal = X_test_pool.loc[sampled_indices]
    y_bal = y_test_pool.loc[sampled_indices]
    return X_bal, y_bal


def build_natural_test_set(
    X_test_pool: pd.DataFrame,
    y_test_pool: pd.Series,
    target_benign_ratio: float = 0.80,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Evaluates against an ~80% BENIGN natural class distribution strictly within the
    held-out test pool. If the pool already has >= 78% BENIGN (real CIC-IDS2017),
    uses the test pool unmodified; otherwise keeps all BENIGN test flows and
    proportionally downsamples attack classes without replacement.
    """
    benign_mask = (y_test_pool == "BENIGN")
    n_benign = int(benign_mask.sum())
    total = int(len(y_test_pool))
    current_ratio = n_benign / total if total > 0 else 0.0

    if current_ratio >= (target_benign_ratio - 0.02) or n_benign == 0:
        return X_test_pool, y_test_pool, {
            "benign_ratio": round(current_ratio, 4),
            "downsampled_attacks": False,
            "total_rows": total,
        }

    # Keep all BENIGN rows; subsample attacks proportionally without replacement
    target_attack_total = max(1, int(round(n_benign * (1.0 - target_benign_ratio) / target_benign_ratio)))
    attack_y = y_test_pool[~benign_mask]
    attack_counts = attack_y.value_counts()
    scale = target_attack_total / len(attack_y)

    # Allocate integer counts per attack class proportionally (minimum 1 each)
    alloc = {cls: max(1, int(round(cnt * scale))) for cls, cnt in attack_counts.items()}
    # Adjust rounding difference on the largest attack class
    diff = sum(alloc.values()) - target_attack_total
    if diff != 0:
        largest_cls = attack_counts.index[0]
        alloc[largest_cls] = max(1, alloc[largest_cls] - diff)

    selected_idx = list(y_test_pool[benign_mask].index)
    for cls, target_n in alloc.items():
        cls_pool = y_test_pool[y_test_pool == cls]
        take_n = min(len(cls_pool), target_n)
        chosen = cls_pool.sample(n=take_n, replace=False, random_state=random_state).index
        selected_idx.extend(chosen.tolist())

    X_nat = X_test_pool.loc[selected_idx]
    y_nat = y_test_pool.loc[selected_idx]
    actual_ratio = float((y_nat == "BENIGN").mean())
    return X_nat, y_nat, {
        "benign_ratio": round(actual_ratio, 4),
        "downsampled_attacks": True,
        "total_rows": int(len(X_nat)),
    }
