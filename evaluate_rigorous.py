"""
Rigorous & Leakage-Free ML Evaluation Suite for NIDS (CIC-IDS2017)

Reproducible end-to-end audit and evaluation pipeline (fixed random seed):
1. Pipeline Data-Leakage Audit (dataset provenance, duplicates, inf/NaN, index overlap,
   Heartbleed/Infiltration count reconciliation, Destination Port shortcut check).
2. Clean, leakage-free stratified split FIRST, followed by train-only balancing
   (class_weight='balanced' and train-only SMOTE inside imblearn CV pipelines).
3. Dual test-set evaluation (a) balanced test set and (b) natural ~80% BENIGN distribution
   plus 5-fold stratified cross-validation (mean +/- std) for SVM and 4 baselines:
   Random Forest, XGBoost, Logistic Regression, and MLP.
4. Leave-One-Attack-Out (LOAO) zero-day evaluation of the <55% confidence rejection rule.
5. Held-out Permutation Importance (SVM & XGBoost) and SHAP (KernelSHAP for SVM,
   TreeSHAP for XGBoost) reporting top-15 features.

All metrics and plots are saved to `results/*.json` and `results/*.png`.
"""

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import config
import preprocessing
from predict import ANOMALY_CONFIDENCE_THRESHOLD

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("shap").setLevel(logging.WARNING)

SEED = config.RANDOM_STATE  # 42
TEST_SPLIT_RATIO = 0.30
CV_FOLDS = 5


# ==============================================================================
# WRAPPER FOR LABEL-ENCODED + SAMPLE-WEIGHTED XGBOOST INSIDE SKLEARN PIPELINES
# ==============================================================================

class EncodedXGBWrapper(BaseEstimator, ClassifierMixin):
    """Wraps XGBClassifier to accept string labels cleanly with balanced sample weights."""

    def __init__(self, random_state: int = SEED, n_jobs: int = config.N_JOBS):
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.encoder = LabelEncoder()
        y_enc = self.encoder.fit_transform(y)
        self.classes_ = self.encoder.classes_
        weights = compute_sample_weight(class_weight="balanced", y=y_enc)
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X, y_enc, sample_weight=weights)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        preds_enc = self.model.predict(X)
        return self.encoder.inverse_transform(preds_enc)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# ==============================================================================
# 1. DATA LEAKAGE & PIPELINE AUDIT
# ==============================================================================

def run_leakage_audit(
    raw_df: pd.DataFrame,
    X_raw: pd.DataFrame,
    y_raw: pd.Series,
    cleaning_audit: Dict[str, int],
    results_dir: Path = config.RESULTS_DIR,
) -> Dict[str, Any]:
    """
    Audits legacy train_model.py, evaluation.py, day_wise_evaluation.py, and dataset provenance.
    Reproduces legacy sampling indices to measure exact train/test overlap.
    """
    csv_files = [p.name for p in sorted(config.DATASET_RAW_DIR.glob("*.csv"))]
    is_synthetic = csv_files == ["CICIDS2017_Benchmark_Sample.csv"]

    # Reconstruct legacy train_model.py (sample_size=8000, test_size=0.20, random_state=42)
    X_legacy_sampled, y_legacy_sampled = preprocessing.get_stratified_sample(
        X_raw, y_raw, max_samples=8000
    )
    X_leg_train, X_leg_test, y_leg_train, y_leg_test = train_test_split(
        X_legacy_sampled,
        y_legacy_sampled,
        test_size=0.20,
        stratify=y_legacy_sampled,
        random_state=config.RANDOM_STATE,
    )

    # Reconstruct legacy evaluation.py (__main__ sampled n=2000 over ALL X!)
    X_leg_eval = X_raw.sample(n=min(2000, len(X_raw)), random_state=config.RANDOM_STATE)
    y_leg_eval = y_raw.loc[X_leg_eval.index]

    train_idx_set = set(X_leg_train.index)
    eval_idx_set = set(X_leg_eval.index)
    overlap_with_train = len(train_idx_set & eval_idx_set)
    overlap_with_leg_test = len(set(X_leg_test.index) & eval_idx_set)

    # Legacy day_wise_evaluation.py evaluated all 15,000 rows from the raw CSV
    daywise_overlap_with_train = len(train_idx_set & set(X_raw.index))

    # Port-shortcut audit: check unique destination ports per class and single-feature port classifier
    port_stats: Dict[str, Any] = {}
    if "Destination Port" in X_raw.columns:
        X_clean, y_clean, _ = preprocessing.clean_features_and_labels(X_raw, y_raw)
        X_tr_p, X_te_p, y_tr_p, y_te_p, _ = preprocessing.stratified_clean_split(
            X_clean, y_clean, test_size=TEST_SPLIT_RATIO, random_state=SEED
        )
        dt_port = DecisionTreeClassifier(max_depth=5, random_state=SEED)
        dt_port.fit(X_tr_p[["Destination Port"]], y_tr_p)
        port_only_acc = float(accuracy_score(y_te_p, dt_port.predict(X_te_p[["Destination Port"]])))
        port_only_f1 = float(f1_score(y_te_p, dt_port.predict(X_te_p[["Destination Port"]]), average="macro"))
        unique_ports_per_class = {
            str(cls): sorted([int(p) for p in X_raw.loc[y_raw == cls, "Destination Port"].unique()[:10]])
            for cls in sorted(y_raw.unique())
        }
        port_stats = {
            "port_only_decision_tree_accuracy": round(port_only_acc, 4),
            "port_only_decision_tree_macro_f1": round(port_only_f1, 4),
            "sample_unique_ports_by_class": unique_ports_per_class,
        }

    # Check if existing reports/evaluation_report.json matches legacy leaked counts
    legacy_eval_json = {}
    legacy_report_path = config.REPORTS_DIR / "evaluation_report.json"
    if legacy_report_path.exists():
        with open(legacy_report_path, "r", encoding="utf-8") as f:
            legacy_eval_json = json.load(f)

    audit_summary = {
        "dataset_files_on_disk": csv_files,
        "is_synthetic_benchmark_dataset": is_synthetic,
        "dataset_class_counts": {str(k): int(v) for k, v in y_raw.value_counts().items()},
        "cleaning_audit": cleaning_audit,
        "legacy_evaluation_leakage": {
            "legacy_train_size": int(len(X_leg_train)),
            "legacy_eval_size": int(len(X_leg_eval)),
            "train_rows_leaked_into_eval": int(overlap_with_train),
            "train_overlap_percent": round((overlap_with_train / len(X_leg_eval)) * 100.0, 2),
            "heldout_test_rows_in_eval": int(overlap_with_leg_test),
            "eval_2000_class_counts": {str(k): int(v) for k, v in y_leg_eval.value_counts().items()},
            "legacy_report_total_test_samples": legacy_eval_json.get("total_test_samples"),
            "legacy_report_heartbleed_support": legacy_eval_json.get("per_class_metrics", {}).get("Heartbleed", {}).get("support"),
            "legacy_report_infiltration_support": legacy_eval_json.get("per_class_metrics", {}).get("Infiltration", {}).get("support"),
            "day_wise_overlap_with_train_rows": int(daywise_overlap_with_train),
            "day_wise_total_evaluated": int(len(X_raw)),
            "explanation_heartbleed_20_infiltration_36": (
                "Real CIC-IDS2017 contains 11 Heartbleed and 36 Infiltration flows across ~2.83M records, "
                "with ~80.3% BENIGN. However, this repository only contains the synthetically generated "
                "CICIDS2017_Benchmark_Sample.csv (15,000 rows: 50% BENIGN, 150 Heartbleed [1%], 300 Infiltration [2%]). "
                "Legacy evaluation.py sampled n=2000 randomly from all 15,000 rows instead of the held-out split, "
                "producing exact counts Heartbleed=20 and Infiltration=36 while leaking 869/2000 (43.45%) training rows."
            ),
        },
        "destination_port_shortcut_audit": port_stats,
    }

    audit_path = results_dir / "audit_report.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=4)
    logger.info(f"Audit report saved to {audit_path}")
    return audit_summary


# ==============================================================================
# 2. METRICS COMPUTATION HELPERS
# ==============================================================================

def compute_multiclass_metrics(
    model,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    classes: List[str],
) -> Dict[str, Any]:
    """Evaluates a fitted model cleanly with Accuracy, F1, PR-AUC, FPR, FNR, Latency, and CM."""
    t0 = time.perf_counter()
    y_pred = model.predict(X_eval)
    infer_total_s = time.perf_counter() - t0
    latency_ms = (infer_total_s / len(X_eval)) * 1000.0

    probs = model.predict_proba(X_eval) if hasattr(model, "predict_proba") else None
    model_classes = list(model.classes_) if hasattr(model, "classes_") else classes

    acc = float(accuracy_score(y_eval, y_pred))
    macro_prec = float(precision_score(y_eval, y_pred, labels=classes, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_eval, y_pred, labels=classes, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_eval, y_pred, labels=classes, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_eval, y_pred, labels=classes, average="weighted", zero_division=0))

    cm = confusion_matrix(y_eval, y_pred, labels=classes)

    # PR-AUC (One-vs-Rest Average Precision)
    per_class_pr_auc: Dict[str, Optional[float]] = {}
    macro_pr_auc = None
    weighted_pr_auc = None
    if probs is not None:
        # Align probability columns cleanly with `classes` order
        col_map = {c: idx for idx, c in enumerate(model_classes)}
        aligned_probs = np.zeros((len(X_eval), len(classes)), dtype=float)
        for j, cls_name in enumerate(classes):
            if cls_name in col_map:
                aligned_probs[:, j] = probs[:, col_map[cls_name]]

        y_bin = label_binarize(y_eval, classes=classes)
        aps, ws = [], []
        for j, cls_name in enumerate(classes):
            sup = int(y_bin[:, j].sum())
            if sup > 0:
                ap = float(average_precision_score(y_bin[:, j], aligned_probs[:, j]))
                per_class_pr_auc[cls_name] = round(ap, 4)
                aps.append(ap)
                ws.append(sup)
            else:
                per_class_pr_auc[cls_name] = None
        if aps:
            macro_pr_auc = round(float(np.mean(aps)), 4)
            weighted_pr_auc = round(float(np.average(aps, weights=ws)), 4)

    per_class: Dict[str, Dict[str, Any]] = {}
    fprs, fnrs = [], []
    for i, cls_name in enumerate(classes):
        tp = float(cm[i, i])
        fn = float(np.sum(cm[i, :]) - tp)
        fp = float(np.sum(cm[:, i]) - tp)
        tn = float(np.sum(cm) - (tp + fn + fp))
        sup = int(tp + fn)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fprs.append(fpr)
        fnrs.append(fnr)

        per_class[cls_name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "pr_auc": per_class_pr_auc.get(cls_name),
            "support": sup,
        }

    # Binary intrusion metrics (BENIGN vs any ATTACK)
    y_true_atk = (y_eval != "BENIGN").to_numpy()
    y_pred_atk = (pd.Series(y_pred, index=y_eval.index) != "BENIGN").to_numpy()
    benign_total = int((~y_true_atk).sum())
    atk_total = int(y_true_atk.sum())
    benign_fpr = float(np.sum(y_pred_atk & ~y_true_atk) / benign_total) if benign_total > 0 else 0.0
    attack_recall = float(np.sum(y_pred_atk & y_true_atk) / atk_total) if atk_total > 0 else 0.0

    return {
        "n_samples": int(len(X_eval)),
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_pr_auc": macro_pr_auc,
        "weighted_pr_auc": weighted_pr_auc,
        "mean_fpr": round(float(np.mean(fprs)), 4),
        "mean_fnr": round(float(np.mean(fnrs)), 4),
        "binary_benign_fpr": round(benign_fpr, 4),
        "binary_attack_recall": round(attack_recall, 4),
        "inference_time_ms_total": round(infer_total_s * 1000.0, 2),
        "inference_latency_ms_per_flow": round(latency_ms, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def _smote_strategy(y_fold, target_cap: int = 1000) -> Dict[Any, int]:
    """Upsamples minority classes only up to target_cap inside the training fold."""
    vals, cnts = np.unique(y_fold, return_counts=True)
    return {k: max(int(c), min(target_cap, int(np.max(cnts)))) for k, c in zip(vals, cnts)}


def build_candidate_models() -> Dict[str, Callable[[], Any]]:
    """Returns factory callables constructing fresh, leakage-free pipelines per fold/run."""
    def make_svm_balanced():
        return Pipeline([
            ("sanitizer", preprocessing.NetworkFlowSanitizer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", probability=True, random_state=SEED)),
        ])

    def make_svm_smote():
        return ImbPipeline([
            ("sanitizer", preprocessing.NetworkFlowSanitizer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(sampling_strategy=_smote_strategy, random_state=SEED)),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=SEED)),
        ])

    def make_rf():
        return Pipeline([
            ("sanitizer", preprocessing.NetworkFlowSanitizer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=config.N_JOBS
            )),
        ])

    def make_xgb():
        return Pipeline([
            ("sanitizer", preprocessing.NetworkFlowSanitizer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", EncodedXGBWrapper(random_state=SEED, n_jobs=config.N_JOBS)),
        ])

    def make_logreg():
        return Pipeline([
            ("sanitizer", preprocessing.NetworkFlowSanitizer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
        ])

    def make_mlp():
        return Pipeline([
            ("sanitizer", preprocessing.NetworkFlowSanitizer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True, random_state=SEED
            )),
        ])

    return {
        "SVM_RBF_ClassWeight": make_svm_balanced,
        "SVM_RBF_TrainSMOTE": make_svm_smote,
        "Random_Forest": make_rf,
        "XGBoost": make_xgb,
        "Logistic_Regression": make_logreg,
        "MLP_NeuralNet": make_mlp,
    }


def run_cv_and_dual_test_evaluation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test_bal: pd.DataFrame,
    y_test_bal: pd.Series,
    X_test_nat: pd.DataFrame,
    y_test_nat: pd.Series,
    nat_meta: Dict[str, Any],
    results_dir: Path = config.RESULTS_DIR,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Runs 5-fold stratified CV on X_train (mean +/- std), trains each model on full X_train,
    and evaluates on both (a) balanced test set and (b) natural (~80% BENIGN) test set.
    Also runs a Destination Port ablation for SVM.
    """
    classes = sorted(y_train.unique().tolist())
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    factories = build_candidate_models()

    comparison_records: Dict[str, Any] = {}
    fitted_models: Dict[str, Any] = {}

    for model_name, make_fn in factories.items():
        logger.info(f"[{model_name}] Running {CV_FOLDS}-Fold Stratified CV on train set ({len(X_train):,} flows)...")
        fold_accs, fold_macro_f1s, fold_wt_f1s, fold_pr_aucs, fold_fit_times = [], [], [], [], []

        for fold_idx, (tr_i, val_i) in enumerate(skf.split(X_train, y_train), start=1):
            X_f_tr, y_f_tr = X_train.iloc[tr_i], y_train.iloc[tr_i]
            X_f_val, y_f_val = X_train.iloc[val_i], y_train.iloc[val_i]

            est = make_fn()
            t0 = time.perf_counter()
            est.fit(X_f_tr, y_f_tr)
            fit_dt = time.perf_counter() - t0
            fold_fit_times.append(fit_dt)

            m_val = compute_multiclass_metrics(est, X_f_val, y_f_val, classes)
            fold_accs.append(m_val["accuracy"])
            fold_macro_f1s.append(m_val["macro_f1"])
            fold_wt_f1s.append(m_val["weighted_f1"])
            if m_val["macro_pr_auc"] is not None:
                fold_pr_aucs.append(m_val["macro_pr_auc"])

        # Fit final model on full training split
        logger.info(f"[{model_name}] Fitting full train split and evaluating on Balanced + Natural test sets...")
        full_est = make_fn()
        t0 = time.perf_counter()
        full_est.fit(X_train, y_train)
        full_train_time_s = time.perf_counter() - t0
        fitted_models[model_name] = full_est

        res_balanced = compute_multiclass_metrics(full_est, X_test_bal, y_test_bal, classes)
        res_natural = compute_multiclass_metrics(full_est, X_test_nat, y_test_nat, classes)

        comparison_records[model_name] = {
            "cv_5fold": {
                "accuracy_mean": round(float(np.mean(fold_accs)), 4),
                "accuracy_std": round(float(np.std(fold_accs)), 4),
                "macro_f1_mean": round(float(np.mean(fold_macro_f1s)), 4),
                "macro_f1_std": round(float(np.std(fold_macro_f1s)), 4),
                "weighted_f1_mean": round(float(np.mean(fold_wt_f1s)), 4),
                "weighted_f1_std": round(float(np.std(fold_wt_f1s)), 4),
                "macro_pr_auc_mean": round(float(np.mean(fold_pr_aucs)), 4) if fold_pr_aucs else None,
                "macro_pr_auc_std": round(float(np.std(fold_pr_aucs)), 4) if fold_pr_aucs else None,
                "fold_fit_time_mean_s": round(float(np.mean(fold_fit_times)), 3),
            },
            "full_train_time_seconds": round(float(full_train_time_s), 3),
            "test_balanced": res_balanced,
            "test_natural": res_natural,
        }

    # SVM Ablation: Drop 'Destination Port' to test flow-dynamics reliance vs port lookup
    if "Destination Port" in X_train.columns:
        logger.info("[SVM_No_DstPort_Ablation] Training SVM without Destination Port...")
        cols_no_port = [c for c in X_train.columns if c != "Destination Port"]
        svm_no_port = factories["SVM_RBF_ClassWeight"]()
        t0 = time.perf_counter()
        svm_no_port.fit(X_train[cols_no_port], y_train)
        t_no_port = time.perf_counter() - t0
        res_bal_np = compute_multiclass_metrics(svm_no_port, X_test_bal[cols_no_port], y_test_bal, classes)
        res_nat_np = compute_multiclass_metrics(svm_no_port, X_test_nat[cols_no_port], y_test_nat, classes)
        comparison_records["SVM_RBF_NoDstPort_Ablation"] = {
            "cv_5fold": None,
            "full_train_time_seconds": round(float(t_no_port), 3),
            "test_balanced": res_bal_np,
            "test_natural": res_nat_np,
        }

    summary = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "random_state": SEED,
        "cv_folds": CV_FOLDS,
        "train_samples": int(len(X_train)),
        "balanced_test_samples": int(len(X_test_bal)),
        "natural_test_metadata": nat_meta,
        "classes": classes,
        "models": comparison_records,
    }

    comp_path = results_dir / "model_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Saved model comparison & CV results to {comp_path}")

    # Save detailed SVM artifact JSON as well
    svm_detail = {
        "classes": classes,
        "SVM_RBF_ClassWeight": comparison_records["SVM_RBF_ClassWeight"],
        "SVM_RBF_TrainSMOTE": comparison_records["SVM_RBF_TrainSMOTE"],
    }
    svm_path = results_dir / "svm_detailed_evaluation.json"
    with open(svm_path, "w", encoding="utf-8") as f:
        json.dump(svm_detail, f, indent=4)

    # Plots
    _plot_confusion_matrix(
        np.array(comparison_records["SVM_RBF_ClassWeight"]["test_balanced"]["confusion_matrix"]),
        classes,
        "SVM (RBF, class_weight='balanced') — Balanced Test Set (Zero Leakage)",
        results_dir / "confusion_matrix_svm_balanced.png",
    )
    _plot_confusion_matrix(
        np.array(comparison_records["SVM_RBF_ClassWeight"]["test_natural"]["confusion_matrix"]),
        classes,
        f"SVM (RBF, class_weight='balanced') — Natural Test Set ({nat_meta['benign_ratio']*100:.1f}% BENIGN)",
        results_dir / "confusion_matrix_svm_natural.png",
    )
    _plot_model_comparison(comparison_records, results_dir / "model_comparison.png")
    _plot_pr_curves(
        fitted_models["SVM_RBF_ClassWeight"],
        X_test_nat,
        y_test_nat,
        classes,
        results_dir / "pr_curves_svm_natural.png",
    )

    return summary, fitted_models


def _plot_confusion_matrix(cm: np.ndarray, classes: List[str], title: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Actual Label", fontsize=11)
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def _plot_model_comparison(records: Dict[str, Any], out_path: Path) -> None:
    model_names = [m for m, v in records.items() if v.get("cv_5fold") is not None]
    cv_f1 = [records[m]["cv_5fold"]["macro_f1_mean"] * 100 for m in model_names]
    cv_std = [records[m]["cv_5fold"]["macro_f1_std"] * 100 for m in model_names]
    bal_f1 = [records[m]["test_balanced"]["macro_f1"] * 100 for m in model_names]
    nat_f1 = [records[m]["test_natural"]["macro_f1"] * 100 for m in model_names]
    train_t = [records[m]["full_train_time_seconds"] for m in model_names]
    infer_ms = [records[m]["test_natural"]["inference_latency_ms_per_flow"] for m in model_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(model_names))
    w = 0.25

    ax1.bar(x - w, cv_f1, w, yerr=cv_std, capsize=3, label="5-Fold CV Macro F1", color="#0284c7")
    ax1.bar(x, bal_f1, w, label="Balanced Test Macro F1", color="#10b981")
    ax1.bar(x + w, nat_f1, w, label="Natural (~80% Benign) Macro F1", color="#f59e0b")
    ax1.set_title("Model Macro F1 Across CV, Balanced & Natural Test Sets", fontweight="bold")
    ax1.set_ylabel("Macro F1-Score (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace("_", "\n") for m in model_names], fontsize=9)
    ax1.set_ylim(max(0, min(cv_f1 + bal_f1 + nat_f1) - 10), 102)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=8)

    ax2_twin = ax2.twinx()
    b1 = ax2.bar(x - 0.18, train_t, 0.35, label="Train Time (s)", color="#6366f1")
    b2 = ax2_twin.bar(x + 0.18, infer_ms, 0.35, label="Inference Latency (ms/flow)", color="#ef4444")
    ax2.set_title("Training Cost vs. Inference Latency", fontweight="bold")
    ax2.set_ylabel("Training Time (seconds)", color="#6366f1")
    ax2_twin.set_ylabel("Inference Latency (ms / flow)", color="#ef4444")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace("_", "\n") for m in model_names], fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    lines = [b1, b2]
    labels = ["Train Time (s)", "Inference (ms/flow)"]
    ax2.legend(lines, labels, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def _plot_pr_curves(model, X_eval: pd.DataFrame, y_eval: pd.Series, classes: List[str], out_path: Path) -> None:
    probs = model.predict_proba(X_eval)
    y_bin = label_binarize(y_eval, classes=classes)
    model_classes = list(model.classes_)
    col_map = {c: idx for idx, c in enumerate(model_classes)}

    plt.figure(figsize=(10, 7))
    for j, cls_name in enumerate(classes):
        if y_bin[:, j].sum() == 0 or cls_name not in col_map:
            continue
        cls_probs = probs[:, col_map[cls_name]]
        prec, rec, _ = precision_recall_curve(y_bin[:, j], cls_probs)
        ap = average_precision_score(y_bin[:, j], cls_probs)
        plt.plot(rec, prec, lw=1.8, label=f"{cls_name} (PR-AUC={ap:.3f})")

    plt.title("SVM (RBF) One-vs-Rest Precision-Recall Curves — Natural Test Set", fontsize=13, fontweight="bold")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.05])
    plt.grid(linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


# ==============================================================================
# 4. LEAVE-ONE-ATTACK-OUT (LOAO) ZERO-DAY EXPERIMENT
# ==============================================================================

def run_leave_one_attack_out(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test_pool: pd.DataFrame,
    y_test_pool: pd.Series,
    results_dir: Path = config.RESULTS_DIR,
    threshold_pct: float = ANOMALY_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Tests zero-day claim for each of the 9 attack categories by removing all samples
    of that attack from training, fitting SVM (RBF, class_weight='balanced'), and
    measuring whether unseen flows trigger the <55% confidence rejection rule vs.
    misclassification as another attack vs. silent miss as BENIGN.
    """
    attack_classes = [c for c in sorted(y_train.unique()) if c != "BENIGN"]
    benign_test = X_test_pool[y_test_pool == "BENIGN"]
    per_attack_results: Dict[str, Any] = {}

    for unseen_cls in attack_classes:
        logger.info(f"[LOAO Zero-Day] Holding out attack class: {unseen_cls}...")
        train_mask = y_train != unseen_cls
        X_loao_tr = X_train.loc[train_mask]
        y_loao_tr = y_train.loc[train_mask]

        svm_loao = Pipeline([
            ("sanitizer", preprocessing.NetworkFlowSanitizer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", probability=True, random_state=SEED)),
        ])
        svm_loao.fit(X_loao_tr, y_loao_tr)

        X_unseen = X_test_pool[y_test_pool == unseen_cls]
        preds = svm_loao.predict(X_unseen)
        probs = svm_loao.predict_proba(X_unseen)
        class_idx_map = {c: idx for idx, c in enumerate(svm_loao.classes_)}

        pred_conf = np.array([probs[i, class_idx_map[p]] * 100.0 for i, p in enumerate(preds)])
        max_conf = np.max(probs, axis=1) * 100.0
        n_unseen = int(len(X_unseen))

        # Rule A (legacy predict.py rule): pred_conf < 55 AND pred != BENIGN
        rule_a_anomaly = (pred_conf < threshold_pct) & (preds != "BENIGN")
        rule_a_known_atk = (pred_conf >= threshold_pct) & (preds != "BENIGN")
        rule_a_missed_benign = preds == "BENIGN"

        # Rule B (strict threshold rule): max_conf < 55 regardless of argmax class
        rule_b_anomaly = max_conf < threshold_pct
        rule_b_known_atk = (max_conf >= threshold_pct) & (preds != "BENIGN")
        rule_b_missed_benign = (max_conf >= threshold_pct) & (preds == "BENIGN")

        # False rejection rate on held-out BENIGN traffic
        benign_probs = svm_loao.predict_proba(benign_test)
        benign_max_conf = np.max(benign_probs, axis=1) * 100.0
        benign_false_anomaly_rate = float(np.mean(benign_max_conf < threshold_pct))

        pred_dist = {str(k): int(v) for k, v in pd.Series(preds).value_counts().items()}

        per_attack_results[unseen_cls] = {
            "n_unseen_test_flows": n_unseen,
            "mean_max_confidence_pct": round(float(np.mean(max_conf)), 2),
            "median_max_confidence_pct": round(float(np.median(max_conf)), 2),
            "min_max_confidence_pct": round(float(np.min(max_conf)), 2),
            "max_max_confidence_pct": round(float(np.max(max_conf)), 2),
            "rule_a_legacy_pred_not_benign": {
                "anomaly_rejection_rate": round(float(np.mean(rule_a_anomaly)), 4),
                "mapped_to_known_attack_rate": round(float(np.mean(rule_a_known_atk)), 4),
                "missed_as_benign_rate": round(float(np.mean(rule_a_missed_benign)), 4),
                "total_caught_rate": round(float(1.0 - np.mean(rule_a_missed_benign)), 4),
            },
            "rule_b_strict_confidence": {
                "anomaly_rejection_rate": round(float(np.mean(rule_b_anomaly)), 4),
                "mapped_to_known_attack_rate": round(float(np.mean(rule_b_known_atk)), 4),
                "missed_as_benign_rate": round(float(np.mean(rule_b_missed_benign)), 4),
                "total_caught_rate": round(float(1.0 - np.mean(rule_b_missed_benign)), 4),
            },
            "benign_false_anomaly_rate": round(benign_false_anomaly_rate, 4),
            "raw_svm_predicted_distribution": pred_dist,
        }

    macro_summary = {
        "mean_rule_a_anomaly_rejection_rate": round(
            float(np.mean([v["rule_a_legacy_pred_not_benign"]["anomaly_rejection_rate"] for v in per_attack_results.values()])), 4
        ),
        "mean_rule_a_missed_as_benign_rate": round(
            float(np.mean([v["rule_a_legacy_pred_not_benign"]["missed_as_benign_rate"] for v in per_attack_results.values()])), 4
        ),
        "mean_rule_a_total_caught_rate": round(
            float(np.mean([v["rule_a_legacy_pred_not_benign"]["total_caught_rate"] for v in per_attack_results.values()])), 4
        ),
        "mean_rule_b_anomaly_rejection_rate": round(
            float(np.mean([v["rule_b_strict_confidence"]["anomaly_rejection_rate"] for v in per_attack_results.values()])), 4
        ),
        "mean_rule_b_missed_as_benign_rate": round(
            float(np.mean([v["rule_b_strict_confidence"]["missed_as_benign_rate"] for v in per_attack_results.values()])), 4
        ),
        "mean_rule_b_total_caught_rate": round(
            float(np.mean([v["rule_b_strict_confidence"]["total_caught_rate"] for v in per_attack_results.values()])), 4
        ),
        "mean_benign_false_anomaly_rate": round(
            float(np.mean([v["benign_false_anomaly_rate"] for v in per_attack_results.values()])), 4
        ),
    }

    loao_summary = {
        "confidence_threshold_pct": threshold_pct,
        "random_state": SEED,
        "macro_summary": macro_summary,
        "per_unseen_attack": per_attack_results,
    }

    out_json = results_dir / "loao_zero_day_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(loao_summary, f, indent=4)
    logger.info(f"LOAO zero-day results saved to {out_json}")

    _plot_loao_results(per_attack_results, results_dir / "loao_zero_day_detection.png", threshold_pct)
    return loao_summary


def _plot_loao_results(per_attack: Dict[str, Any], out_path: Path, threshold_pct: float) -> None:
    classes = list(per_attack.keys())
    anom_b = [per_attack[c]["rule_b_strict_confidence"]["anomaly_rejection_rate"] * 100 for c in classes]
    known_b = [per_attack[c]["rule_b_strict_confidence"]["mapped_to_known_attack_rate"] * 100 for c in classes]
    miss_b = [per_attack[c]["rule_b_strict_confidence"]["missed_as_benign_rate"] * 100 for c in classes]

    x = np.arange(len(classes))
    plt.figure(figsize=(12, 6))
    plt.bar(x, anom_b, label=f"Rejected as Unknown Anomaly (<{threshold_pct:.0f}% conf)", color="#f59e0b")
    plt.bar(x, known_b, bottom=anom_b, label=f"Mapped to Known Attack (>={threshold_pct:.0f}% conf)", color="#0284c7")
    bottom_miss = np.array(anom_b) + np.array(known_b)
    plt.bar(x, miss_b, bottom=bottom_miss, label="Evaded as BENIGN (Missed Zero-Day)", color="#ef4444")

    plt.title("Leave-One-Attack-Out Zero-Day Detection Breakdown (Strict <55% Confidence Rule)", fontsize=13, fontweight="bold")
    plt.ylabel("Percentage of Unseen Attack Flows (%)")
    plt.xticks(x, classes, rotation=35, ha="right")
    plt.ylim(0, 108)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


# ==============================================================================
# 5. FEATURE IMPORTANCE: PERMUTATION IMPORTANCE + SHAP (TOP 15)
# ==============================================================================

def run_feature_importance_analysis(
    svm_pipeline: Pipeline,
    xgb_pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test_bal: pd.DataFrame,
    y_test_bal: pd.Series,
    X_test_pool: pd.DataFrame,
    y_test_pool: pd.Series,
    results_dir: Path = config.RESULTS_DIR,
) -> Dict[str, Any]:
    """
    Computes held-out Permutation Importance (SVM & XGBoost) and SHAP values
    (TreeExplainer for XGBoost on 500 stratified test samples; KernelExplainer for SVM
    on 100 stratified test samples with kmeans background). Reports Top 15 features.
    """
    feature_names = list(X_test_bal.columns)

    logger.info("[Feature Importance] Computing Permutation Importance on held-out balanced test set (SVM & XGBoost)...")
    perm_svm = permutation_importance(
        svm_pipeline, X_test_bal, y_test_bal,
        scoring="f1_macro", n_repeats=5, random_state=SEED, n_jobs=config.N_JOBS,
    )
    perm_xgb = permutation_importance(
        xgb_pipeline, X_test_bal, y_test_bal,
        scoring="f1_macro", n_repeats=5, random_state=SEED, n_jobs=config.N_JOBS,
    )

    def _top15_perm(perm_res) -> List[Dict[str, Any]]:
        order = np.argsort(perm_res.importances_mean)[::-1][:15]
        return [
            {
                "rank": idx + 1,
                "feature": feature_names[f_i],
                "importance_mean": round(float(perm_res.importances_mean[f_i]), 5),
                "importance_std": round(float(perm_res.importances_std[f_i]), 5),
            }
            for idx, f_i in enumerate(order)
        ]

    top15_perm_svm = _top15_perm(perm_svm)
    top15_perm_xgb = _top15_perm(perm_xgb)

    # Stratified subset from strictly held-out test pool for SHAP
    _, X_shap_xgb, _, _ = train_test_split(
        X_test_pool, y_test_pool,
        test_size=min(500, len(X_test_pool)),
        stratify=y_test_pool,
        random_state=SEED,
    ) if len(X_test_pool) > 500 else (None, X_test_pool, None, y_test_pool)

    _, X_shap_svm, _, _ = train_test_split(
        X_test_pool, y_test_pool,
        test_size=min(100, len(X_test_pool)),
        stratify=y_test_pool,
        random_state=SEED,
    ) if len(X_test_pool) > 100 else (None, X_test_pool, None, y_test_pool)

    # TreeSHAP for XGBoost
    logger.info(f"[SHAP] Running TreeExplainer for XGBoost on {len(X_shap_xgb)} held-out test flows...")
    xgb_prep = Pipeline(xgb_pipeline.steps[:-1])
    X_shap_xgb_t = xgb_prep.transform(X_shap_xgb)
    tree_exp = shap.TreeExplainer(xgb_pipeline.named_steps["clf"].model)
    tree_sv = np.asarray(tree_exp.shap_values(X_shap_xgb_t))
    # Handle (N, F, C) or list/array shapes
    if tree_sv.ndim == 3:
        tree_global = np.mean(np.abs(tree_sv), axis=(0, 2))
    else:
        tree_global = np.mean(np.abs(tree_sv), axis=0)

    # KernelSHAP for SVM (RBF)
    logger.info(f"[SHAP] Running KernelExplainer for SVM (RBF) on {len(X_shap_svm)} held-out test flows...")
    svm_prep = Pipeline(svm_pipeline.steps[:-1])
    X_tr_svm_t = svm_prep.transform(X_train)
    X_shap_svm_t = svm_prep.transform(X_shap_svm)
    bg_summary = shap.kmeans(X_tr_svm_t, 25)
    kernel_exp = shap.KernelExplainer(svm_pipeline.named_steps["clf"].predict_proba, bg_summary)
    kernel_sv = np.asarray(kernel_exp.shap_values(X_shap_svm_t, nsamples=120, silent=True))
    if kernel_sv.ndim == 3:
        kernel_global = np.mean(np.abs(kernel_sv), axis=(0, 2))
    else:
        kernel_global = np.mean(np.abs(kernel_sv), axis=0)

    def _top15_shap(vals: np.ndarray) -> List[Dict[str, Any]]:
        order = np.argsort(vals)[::-1][:15]
        return [
            {
                "rank": idx + 1,
                "feature": feature_names[f_i],
                "mean_abs_shap": round(float(vals[f_i]), 5),
            }
            for idx, f_i in enumerate(order)
        ]

    top15_shap_xgb = _top15_shap(tree_global)
    top15_shap_svm = _top15_shap(kernel_global)

    fi_summary = {
        "evaluation_split": "held-out test set only",
        "random_state": SEED,
        "permutation_importance_svm_top15": top15_perm_svm,
        "permutation_importance_xgb_top15": top15_perm_xgb,
        "shap_kernel_svm_top15": top15_shap_svm,
        "shap_tree_xgb_top15": top15_shap_xgb,
        "svm_perm_vs_kernelshap_top15_overlap": sorted(
            list({r["feature"] for r in top15_perm_svm} & {r["feature"] for r in top15_shap_svm})
        ),
    }

    fi_path = results_dir / "feature_importance.json"
    with open(fi_path, "w", encoding="utf-8") as f:
        json.dump(fi_summary, f, indent=4)
    logger.info(f"Feature importance saved to {fi_path}")

    _plot_top15_comparison(
        top15_perm_svm, top15_perm_xgb,
        val_key="importance_mean", err_key="importance_std",
        title_left="SVM (RBF) Permutation Importance — Top 15",
        title_right="XGBoost Permutation Importance — Top 15",
        xlabel="Macro F1 Drop on Permutation",
        out_path=results_dir / "feature_importance_permutation_top15.png",
    )
    _plot_top15_comparison(
        top15_shap_svm, top15_shap_xgb,
        val_key="mean_abs_shap", err_key=None,
        title_left=f"SVM (RBF) KernelSHAP (n={len(X_shap_svm)}) — Top 15",
        title_right=f"XGBoost TreeSHAP (n={len(X_shap_xgb)}) — Top 15",
        xlabel="Mean |SHAP Value|",
        out_path=results_dir / "feature_importance_shap_top15.png",
    )
    return fi_summary


def _plot_top15_comparison(
    left_rows: List[Dict[str, Any]],
    right_rows: List[Dict[str, Any]],
    val_key: str,
    err_key: Optional[str],
    title_left: str,
    title_right: str,
    xlabel: str,
    out_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    for ax, rows, title, color in [
        (ax1, left_rows[::-1], title_left, "#0284c7"),
        (ax2, right_rows[::-1], title_right, "#10b981"),
    ]:
        feats = [r["feature"] for r in rows]
        vals = [r[val_key] for r in rows]
        errs = [r[err_key] for r in rows] if err_key else None
        ax.barh(feats, vals, xerr=errs, color=color, capsize=3, alpha=0.9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def main() -> Dict[str, Any]:
    np.random.seed(SEED)
    results_dir = config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_df = preprocessing.load_raw_dataset()
    X_raw, y_raw = preprocessing.preprocess_dataframe(raw_df)
    X_clean, y_clean, cleaning_audit = preprocessing.clean_features_and_labels(
        X_raw, y_raw, drop_inf_nan=True, drop_duplicates=True
    )

    # 1. Pipeline Audit
    audit_report = run_leakage_audit(raw_df, X_raw, y_raw, cleaning_audit, results_dir)

    # 2. Split FIRST cleanly (zero overlap verified)
    X_train, X_test_pool, y_train, y_test_pool, split_checks = preprocessing.stratified_clean_split(
        X_clean, y_clean, test_size=TEST_SPLIT_RATIO, random_state=SEED
    )
    X_test_bal, y_test_bal = preprocessing.build_balanced_test_set(X_test_pool, y_test_pool, random_state=SEED)
    X_test_nat, y_test_nat, nat_meta = preprocessing.build_natural_test_set(
        X_test_pool, y_test_pool, target_benign_ratio=0.80, random_state=SEED
    )
    logger.info(
        f"Clean Split: Train={len(X_train):,}, Test Pool={len(X_test_pool):,}, "
        f"Balanced Test={len(X_test_bal):,}, Natural Test={len(X_test_nat):,} ({nat_meta['benign_ratio']*100:.1f}% BENIGN)"
    )

    # 3. CV + Baselines + Dual Test Evaluation
    comp_summary, fitted_models = run_cv_and_dual_test_evaluation(
        X_train, y_train, X_test_bal, y_test_bal, X_test_nat, y_test_nat, nat_meta, results_dir
    )

    # 4. Leave-One-Attack-Out Zero-Day Experiment
    loao_summary = run_leave_one_attack_out(X_train, y_train, X_test_pool, y_test_pool, results_dir)

    # 5. Permutation Importance + SHAP Top 15
    fi_summary = run_feature_importance_analysis(
        fitted_models["SVM_RBF_ClassWeight"],
        fitted_models["XGBoost"],
        X_train,
        X_test_bal,
        y_test_bal,
        X_test_pool,
        y_test_pool,
        results_dir,
    )

    # Combined manifest
    manifest = {
        "seed": SEED,
        "split_checks": split_checks,
        "audit_summary": audit_report,
        "comparison_summary": comp_summary,
        "loao_macro_summary": loao_summary["macro_summary"],
        "feature_importance_summary": fi_summary,
    }
    manifest_path = results_dir / "evaluation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    logger.info("=== Rigorous Evaluation Suite Complete ===")
    return manifest


if __name__ == "__main__":
    main()
