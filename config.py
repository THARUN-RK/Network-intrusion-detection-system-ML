"""
Central Configuration for Network Intrusion Detection System (NIDS)
Hardware-optimized for Intel Core i3-1215U / 8 GB RAM / Windows 64-bit
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base Paths
BASE_DIR = Path(__file__).resolve().parent

_PLACEHOLDER_KEYS = {
    "",
    "your_api_key_here",
    "your_virustotal_api_key",
    "your_abuseipdb_api_key",
    "changeme",
    "placeholder",
    "none",
    "null",
}


def load_environment(env_path: Path = None) -> bool:
    """
    Loads local .env variables safely without overriding existing OS / Render environment variables.
    Uses python-dotenv when installed, with a lightweight stdlib parser fallback.
    """
    target_env = env_path or (BASE_DIR / ".env")
    if not target_env.exists() or not target_env.is_file():
        return False

    try:
        from dotenv import load_dotenv
        return bool(load_dotenv(dotenv_path=target_env, override=False))
    except ImportError:
        loaded = False
        try:
            with open(target_env, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    if line.startswith("export "):
                        line = line[len("export "):].strip()
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
                        val = val[1:-1]
                    if key and key not in os.environ:
                        os.environ[key] = val
                        loaded = True
        except Exception:
            return False
        return loaded


# Load local .env automatically if present (Render env vars take precedence via override=False)
DOTENV_LOADED = load_environment()


_ENV_VAR_ALIASES = {
    "VIRUSTOTAL_API_KEY": ("VIRUSTOTAL_API_KEY", "VirusTotalAPI", "VIRUSTOTAL_API", "VT_API_KEY"),
    "ABUSEIPDB_API_KEY": ("ABUSEIPDB_API_KEY", "AbuseIPDBAPI", "ABUSEIPDB_API", "ABUSE_IPDB_API_KEY"),
}


def get_api_key(env_var_name: str) -> str:
    """
    Reads an API key dynamically from the environment (including common fallback aliases),
    stripping whitespace/quotes and rejecting empty or template placeholder strings.
    Never logs or returns secrets.
    """
    candidates = _ENV_VAR_ALIASES.get(env_var_name, (env_var_name,))
    for candidate in candidates:
        raw = os.getenv(candidate, "")
        if not isinstance(raw, str):
            continue
        cleaned = raw.strip().strip('"').strip("'").strip()
        if cleaned and cleaned.lower() not in _PLACEHOLDER_KEYS:
            return cleaned
    return ""


def get_threat_intel_config_status() -> Dict[str, Any]:
    """Returns safe configuration flags for threat intelligence APIs without exposing keys."""
    vt_configured = bool(get_api_key("VIRUSTOTAL_API_KEY"))
    abuse_configured = bool(get_api_key("ABUSEIPDB_API_KEY"))
    return {
        "dotenv_file_present": (BASE_DIR / ".env").exists(),
        "virustotal": {
            "provider": "VirusTotal API v3",
            "env_var": "VIRUSTOTAL_API_KEY",
            "configured": vt_configured,
            "status": "CONFIGURED" if vt_configured else "NOT_CONFIGURED",
        },
        "abuseipdb": {
            "provider": "AbuseIPDB API v2",
            "env_var": "ABUSEIPDB_API_KEY",
            "configured": abuse_configured,
            "status": "CONFIGURED" if abuse_configured else "NOT_CONFIGURED",
        },
    }


DATASET_RAW_DIR = BASE_DIR / "dataset" / "raw"
DATASET_PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Database Configuration
DATABASE_PATH = LOGS_DIR / "predictions.db"

# Model Artifacts
MODEL_PATH = MODELS_DIR / "svm_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# API & Server Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
API_URL = os.getenv("NIDS_API_URL", f"http://127.0.0.1:{API_PORT}")

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
