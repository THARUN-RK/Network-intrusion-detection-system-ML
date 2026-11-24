"""
Central Configuration for Network Intrusion Detection System (NIDS)
Hardware-optimized for Intel Core i3-1215U / 8 GB RAM / Windows 64-bit
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_RAW_DIR = BASE_DIR / "dataset" / "raw"
DATASET_PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

# Database Configuration
DATABASE_PATH = LOGS_DIR / "predictions.db"

# Model Artifacts
MODEL_PATH = MODELS_DIR / "svm_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# API & Server Configuration
API_HOST = "127.0.0.1"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}"

# Hardware Optimization (8GB RAM, i3 6-core)
# SVM computational complexity scales with O(N^2) to O(N^3) with sample size N.
# We limit stratified sample sizes during development to prevent out-of-memory (OOM) paging.
DEFAULT_SAMPLE_SIZE = 50000        # Recommended maximum for 8GB RAM development
SVM_TRAIN_SAMPLE_SIZE = 25000      # Optimized for GridSearch CV with RBF kernel
TEST_SPLIT_RATIO = 0.20
RANDOM_STATE = 42
N_JOBS = 4                         # Avoid saturating all 6 cores to keep system responsive

# Canonical CIC-IDS2017 Attack Categories
CANONICAL_ATTACK_CATEGORIES = [
    "BENIGN",
    "DoS",
    "DDoS",
    "PortScan",
    "Bot",
    "FTP-Patator",
    "SSH-Patator",
    "Web Attack",
    "Infiltration",
    "Heartbleed"
]

# High-variance Informative Features for NIDS Flow Classification
# Flow ID, IP addresses, Timestamps are excluded to prevent temporal and topological data leakage.
CORE_FLOW_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min"
]
