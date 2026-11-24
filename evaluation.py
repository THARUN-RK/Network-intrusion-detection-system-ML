"""
Network Intrusion Detection System (NIDS) - Model Evaluation Module
Calculates comprehensive cybersecurity metrics: Accuracy, Precision, Recall,
Macro F1, FPR, FNR, Latency, and produces professional charts and JSON reports.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_nids_model(
    model_pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path = config.REPORTS_DIR
) -> Dict[str, Any]:
    """
    Evaluates the SVM model against unseen test network flows.
    Computes all standard SOC evaluation metrics and generates reports.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Evaluating model on {len(X_test):,} unseen test records...")
    
    # Measure latency
    start_time = time.perf_counter()
    y_pred = model_pipeline.predict(X_test)
    total_eval_time = time.perf_counter() - start_time
    latency_ms_per_record = (total_eval_time / len(X_test)) * 1000.0
    
    # Classes
    classes = list(model_pipeline.classes_)
    
    # Core Metrics
    acc = float(accuracy_score(y_test, y_pred))
    macro_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    macro_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    
    # Classification Report
    cls_report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cls_report_str = classification_report(y_test, y_pred, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    # Calculate Per-Class FPR and FNR
    per_class_metrics = {}
    for i, cls_name in enumerate(classes):
        tp = float(cm[i, i])
        fn = float(np.sum(cm[i, :]) - tp)
        fp = float(np.sum(cm[:, i]) - tp)
        tn = float(np.sum(cm) - (tp + fn + fp))
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        per_class_metrics[cls_name] = {
            "precision": float(cls_report_dict.get(cls_name, {}).get("precision", 0.0)),
            "recall": float(cls_report_dict.get(cls_name, {}).get("recall", 0.0)),
            "f1_score": float(cls_report_dict.get(cls_name, {}).get("f1-score", 0.0)),
            "support": int(cls_report_dict.get(cls_name, {}).get("support", 0)),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4)
        }
        
    metrics_summary = {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "weighted_precision": round(weighted_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "weighted_recall": round(weighted_rec, 4),
        "macro_f1_score": round(macro_f1, 4),
        "weighted_f1_score": round(weighted_f1, 4),
        "prediction_latency_ms_per_record": round(latency_ms_per_record, 3),
        "total_test_samples": int(len(X_test)),
        "classes": classes,
        "per_class_metrics": per_class_metrics
    }
    
    # 1. Save classification report text
    report_txt_path = output_dir / "classification_report.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("=== NETWORK INTRUSION DETECTION SYSTEM (NIDS) - SVM EVALUATION REPORT ===\n\n")
        f.write(f"Total Test Records: {len(X_test):,}\n")
        f.write(f"Overall Accuracy:   {acc*100:.2f}%\n")
        f.write(f"Macro F1-Score:     {macro_f1*100:.2f}%\n")
        f.write(f"Weighted F1-Score:  {weighted_f1*100:.2f}%\n")
        f.write(f"Prediction Latency: {latency_ms_per_record:.3f} ms / flow\n\n")
        f.write(cls_report_str)
    logger.info(f"Classification report saved to: {report_txt_path}")
    
    # 2. Save JSON report
    report_json_path = output_dir / "evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    logger.info(f"Evaluation metrics JSON saved to: {report_json_path}")
    
    # 3. Generate Confusion Matrix Heatmap
    cm_png_path = output_dir / "confusion_matrix.png"
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title("NIDS Multiclass SVM - Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("Actual Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_png_path, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix visualization saved to: {cm_png_path}")
    
    # 4. Generate Class-wise Performance Chart
    class_perf_png = output_dir / "class_performance.png"
    df_perf = pd.DataFrame(per_class_metrics).T[["precision", "recall", "f1_score"]]
    plt.figure(figsize=(12, 6))
    df_perf.plot(kind="bar", figsize=(12, 6), colormap="viridis")
    plt.title("NIDS Per-Class Precision, Recall, and F1-Score", fontsize=14, fontweight="bold")
    plt.xlabel("Attack Category", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(["Precision", "Recall", "F1-Score"], loc="lower right")
    plt.tight_layout()
    plt.savefig(class_perf_png, dpi=300)
    plt.close()
    logger.info(f"Per-class performance visualization saved to: {class_perf_png}")
    
    return metrics_summary


if __name__ == "__main__":
    if config.MODEL_PATH.exists():
        logger.info(f"Loading existing model from {config.MODEL_PATH} for evaluation...")
        pipeline = joblib.load(config.MODEL_PATH)
        import preprocessing
        raw_df = preprocessing.load_raw_dataset()
        X, y = preprocessing.preprocess_dataframe(raw_df)
        X_test = X.sample(n=min(2000, len(X)), random_state=config.RANDOM_STATE)
        y_test = y.loc[X_test.index]
        evaluate_nids_model(pipeline, X_test, y_test)
    else:
        logger.warning("Model file not found. Run train_model.py first.")
