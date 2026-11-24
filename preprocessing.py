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
