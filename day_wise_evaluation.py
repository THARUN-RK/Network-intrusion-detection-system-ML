"""
Network Intrusion Detection System (NIDS) - Day-Wise Cross-Testing & Temporal Evaluation
Evaluates model generalization across the 5 attack days of the CIC-IDS2017 dataset:
- Monday: Normal Benign baseline traffic
- Tuesday: Brute Force (FTP-Patator, SSH-Patator)
- Wednesday: DoS (Hulk, GoldenEye, Slowloris) & Heartbleed
- Thursday: Web Attacks (SQLi, XSS, Brute Force) & Infiltration
- Friday: Botnet, PortScan, and DDoS
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import config
import preprocessing
from predict import ANOMALY_CONFIDENCE_THRESHOLD

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Canonical CIC-IDS2017 Day Mapping Definition
DAY_DEFINITIONS = {
    "Monday": {
        "description": "Baseline Normal Traffic (No Attacks)",
        "attacks": [],
        "benign_ratio": 1.0
    },
    "Tuesday": {
        "description": "Brute Force Authentication Attacks",
        "attacks": ["FTP-Patator", "SSH-Patator"],
        "benign_ratio": 0.5
    },
    "Wednesday": {
        "description": "Volumetric Denial of Service & Heartbleed",
        "attacks": ["DoS", "Heartbleed"],
        "benign_ratio": 0.4
    },
    "Thursday": {
        "description": "Web Exploitation & Infiltration",
        "attacks": ["Web Attack", "Infiltration"],
        "benign_ratio": 0.5
    },
    "Friday": {
        "description": "Distributed Denial of Service, Botnet & PortScan",
        "attacks": ["DDoS", "PortScan", "Bot"],
        "benign_ratio": 0.35
    }
}


def partition_dataset_by_days(df: pd.DataFrame, random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Partitions dataset records into canonical CIC-IDS2017 day profiles based on
    ground-truth attack taxonomy and representative background benign traffic.
    """
    day_splits = {}
    benign_df = df[df["Label"] == "BENIGN"].copy()
    
    # Shuffle benign once
    benign_shuffled = benign_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    num_days = len(DAY_DEFINITIONS)
    chunk_size = len(benign_shuffled) // num_days
    benign_chunks = [benign_shuffled.iloc[i * chunk_size : (i + 1) * chunk_size if i < num_days - 1 else len(benign_shuffled)] for i in range(num_days)]
    
    for idx, (day_name, day_meta) in enumerate(DAY_DEFINITIONS.items()):
        day_attacks = day_meta["attacks"]
        attack_records = df[df["Label"].isin(day_attacks)].copy() if day_attacks else pd.DataFrame()
        day_benign = benign_chunks[idx].copy()
        
        # Combine attacks with that day's background traffic
        day_combined = pd.concat([attack_records, day_benign], ignore_index=True)
        day_combined = day_combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        day_splits[day_name] = day_combined
        logger.info(f"Day {day_name}: {len(day_combined):,} records ({len(attack_records):,} attacks, {len(day_benign):,} benign)")
        
    return day_splits


def evaluate_day_wise(
    model_pipeline=None,
    dataset_path: Optional[Path] = None,
    output_dir: Path = config.REPORTS_DIR
) -> Dict[str, Any]:
    """
    Executes cross-day temporal evaluation across Monday through Friday.
    Computes accuracy, FPR, attack recall, anomaly rejection, and latency per day.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if model_pipeline is None:
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {config.MODEL_PATH}")
        logger.info(f"Loading trained SVM pipeline from {config.MODEL_PATH}")
        model_pipeline = joblib.load(config.MODEL_PATH)
        
    data_file = dataset_path or (config.DATASET_RAW_DIR / "CICIDS2017_Benchmark_Sample.csv")
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset benchmark not found at {data_file}")
        
    logger.info(f"Reading benchmark dataset from {data_file}...")
    raw_df = pd.read_csv(data_file)
    raw_df["Label"] = raw_df["Label"].apply(preprocessing.canonicalize_label)
    
    day_partitions = partition_dataset_by_days(raw_df, random_state=config.RANDOM_STATE)
    
    day_results = {}
    overall_start = time.perf_counter()
    
    for day_name, day_df in day_partitions.items():
        logger.info(f"Evaluating {day_name} traffic profile...")
        
        X_day = day_df.drop(columns=["Label"], errors="ignore")
        y_day = day_df["Label"]
        
        # Latency check
        t0 = time.perf_counter()
        y_pred = model_pipeline.predict(X_day)
        lat_ms = ((time.perf_counter() - t0) / len(X_day)) * 1000.0
        
        # Check decision confidence or probabilities if available
        is_unknown_anom = np.zeros(len(y_pred), dtype=bool)
        if hasattr(model_pipeline, "predict_proba"):
            probs = model_pipeline.predict_proba(X_day)
            max_probs = np.max(probs, axis=1) * 100.0
            is_unknown_anom = max_probs < ANOMALY_CONFIDENCE_THRESHOLD
            
        acc = float(accuracy_score(y_day, y_pred))
        macro_f1 = float(f1_score(y_day, y_pred, average="macro", zero_division=0))
        
        # Benign False Positive Rate (FPR)
        benign_mask = (y_day == "BENIGN")
        if benign_mask.sum() > 0:
            benign_pred = y_pred[benign_mask]
            fp_count = int(np.sum(benign_pred != "BENIGN"))
            fpr = float(fp_count / benign_mask.sum())
        else:
            fpr = 0.0
            
        # Attack Recall (Detection Rate)
        attack_mask = (y_day != "BENIGN")
        if attack_mask.sum() > 0:
            attack_true = y_day[attack_mask]
            attack_pred = y_pred[attack_mask]
            # True positives for attack detection: any attack detected as an attack
            tp_attacks = int(np.sum(attack_pred != "BENIGN"))
            attack_recall = float(tp_attacks / attack_mask.sum())
        else:
            attack_recall = 1.0  # Monday has no attacks
            
        day_results[day_name] = {
            "description": DAY_DEFINITIONS[day_name]["description"],
            "total_records": int(len(day_df)),
            "attack_records": int(attack_mask.sum()),
            "benign_records": int(benign_mask.sum()),
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "false_positive_rate": round(fpr, 4),
            "attack_detection_rate": round(attack_recall, 4),
            "anomaly_rejections": int(np.sum(is_unknown_anom)),
            "inference_latency_ms": round(lat_ms, 3),
            "attack_types_present": DAY_DEFINITIONS[day_name]["attacks"]
        }
        
    total_duration = time.perf_counter() - overall_start
    
    summary = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_days_evaluated": len(day_results),
        "total_flows_evaluated": sum(d["total_records"] for d in day_results.values()),
        "overall_evaluation_duration_seconds": round(total_duration, 2),
        "anomaly_rejection_threshold_percent": ANOMALY_CONFIDENCE_THRESHOLD,
        "days": day_results
    }
    
    # 1. Save JSON report
    json_path = output_dir / "day_wise_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Day-wise evaluation report saved to: {json_path}")
    
    # 2. Generate Multi-panel Visualization
    try:
        plot_day_wise_summary(summary, output_dir / "day_wise_evaluation.png")
    except Exception as e:
        logger.warning(f"Could not generate day-wise plot: {e}")
        
    return summary


def plot_day_wise_summary(summary: Dict[str, Any], output_path: Path):
    """Generates a professional 4-panel figure illustrating day-wise temporal performance."""
    days = list(summary["days"].keys())
    accs = [summary["days"][d]["accuracy"] * 100.0 for d in days]
    f1s = [summary["days"][d]["macro_f1"] * 100.0 for d in days]
    fprs = [summary["days"][d]["false_positive_rate"] * 100.0 for d in days]
    recalls = [summary["days"][d]["attack_detection_rate"] * 100.0 for d in days]
    attacks = [summary["days"][d]["attack_records"] for d in days]
    benigns = [summary["days"][d]["benign_records"] for d in days]
    latencies = [summary["days"][d]["inference_latency_ms"] for d in days]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CIC-IDS2017 Day-Wise Temporal Generalization & Performance Stability", fontsize=15, fontweight="bold", y=0.98)
    
    # Panel 1: Accuracy & Macro F1
    x = np.arange(len(days))
    width = 0.35
    ax1.bar(x - width/2, accs, width, label="Accuracy (%)", color="#0284c7")
    ax1.bar(x + width/2, f1s, width, label="Macro F1 (%)", color="#10b981")
    ax1.set_title("Classification Accuracy & Macro F1 Across Days", fontweight="bold")
    ax1.set_ylabel("Score (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(days, fontweight="bold")
    ax1.set_ylim(85, 102)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")
    
    # Panel 2: FPR vs Attack Detection Rate
    ax2.plot(days, recalls, marker="o", linewidth=2.5, color="#10b981", label="Attack Detection Rate (%)")
    ax2.plot(days, fprs, marker="s", linewidth=2.5, color="#ef4444", label="False Positive Rate (FPR %)")
    ax2.set_title("Detection Rate vs. False Positive Rate (FPR)", fontweight="bold")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_ylim(-1, 105)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="center right")
    
    # Panel 3: Traffic Composition
    ax3.bar(days, benigns, label="Normal Flows (BENIGN)", color="#64748b", alpha=0.85)
    ax3.bar(days, attacks, bottom=benigns, label="Attack Flows", color="#f97316", alpha=0.9)
    ax3.set_title("Network Flow Composition Per Day", fontweight="bold")
    ax3.set_ylabel("Flow Count")
    ax3.grid(axis="y", linestyle="--", alpha=0.6)
    ax3.legend(loc="upper right")
    
    # Panel 4: Latency per Flow
    ax4.bar(days, latencies, color="#8b5cf6", width=0.45)
    ax4.set_title("Inference Latency Per Flow Across Days", fontweight="bold")
    ax4.set_ylabel("Latency (ms / flow)")
    ax4.grid(axis="y", linestyle="--", alpha=0.6)
    for i, v in enumerate(latencies):
        ax4.text(i, v + 0.005, f"{v:.3f} ms", ha="center", fontsize=9, fontweight="bold")
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Day-wise evaluation visualization saved to: {output_path}")


if __name__ == "__main__":
    evaluate_day_wise()
