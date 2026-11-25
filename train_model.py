"""
Network Intrusion Detection System (NIDS) - SVM Model Training Module
Optimized for Multiclass Attack Detection with RBF Kernel & 8GB RAM Safety Limits.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, f1_score

import config
import preprocessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_svm_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune_hyperparameters: bool = True,
    cv_folds: int = 3
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Constructs and fits an end-to-end Pipeline:
    1. Preprocessing (Sanitizer + Imputer + Scaler)
    2. Support Vector Classifier with RBF Kernel & Class Weight Balancing
    
    Uses GridSearchCV optimized for Macro F1-Score to prioritize detection
    of rare attack categories (e.g., Infiltration, Heartbleed).
    """
    logger.info("Initializing SVM Model Training Pipeline...")
    start_time = time.time()
    
    # 1. Base pipeline definition
    prep_pipeline = preprocessing.build_feature_pipeline()
    
    # SVC with RBF kernel, balanced weights, and probability calibration enabled
    svc = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=config.RANDOM_STATE
    )
    
    full_pipeline = Pipeline([
        ("preprocessor", prep_pipeline),
        ("classifier", svc)
    ])
    
    best_params = {"classifier__C": 10, "classifier__gamma": "scale"}
    cv_score = 0.0
    
    if tune_hyperparameters:
        logger.info(f"Running GridSearchCV with Stratified {cv_folds}-Fold CV...")
        param_grid = {
            "classifier__C": [1, 10],
            "classifier__gamma": ["scale", 0.01]
        }
        
        stratified_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
        grid_search = GridSearchCV(
            estimator=full_pipeline,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=stratified_cv,
            n_jobs=config.N_JOBS,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_score = float(grid_search.best_score_)
        logger.info(f"GridSearchCV Best Params: {best_params}, Best Macro F1: {cv_score:.4f}")
    else:
        logger.info("Fitting base SVM pipeline directly...")
        full_pipeline.set_params(**best_params)
        full_pipeline.fit(X_train, y_train)
        best_pipeline = full_pipeline
        
    training_duration = time.time() - start_time
    logger.info(f"Training completed in {training_duration:.2f} seconds.")
    
    metadata = {
        "model_type": "Support Vector Machine (SVC)",
        "kernel": "rbf",
        "best_params": best_params,
        "cv_folds": cv_folds,
        "cv_macro_f1": cv_score,
        "training_duration_seconds": round(training_duration, 2),
        "feature_count": int(X_train.shape[1]),
        "feature_names": list(X_train.columns),
        "classes": list(best_pipeline.classes_),
        "training_samples_count": int(len(X_train))
    }
    
    return best_pipeline, metadata


def save_model_artifacts(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    model_path: Path = config.MODEL_PATH,
    metadata_path: Path = config.METADATA_PATH
) -> None:
    """Serializes the trained pipeline and metadata safely to disk."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save pipeline binary using joblib
    joblib.dump(pipeline, model_path, compress=3)
    logger.info(f"Model pipeline persisted to: {model_path}")
    
    # Save metadata JSON for API and Streamlit consumption
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Model metadata persisted to: {metadata_path}")


def run_training_workflow(
    sample_size: int = config.SVM_TRAIN_SAMPLE_SIZE,
    tune_grid: bool = True,
    cv_folds: int = 5
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame, pd.Series]:
    """
    Executes the complete leakage-free training workflow:
    1. Loads dataset
    2. Preprocesses, canonicalizes labels, and drops inf/NaN and duplicate rows BEFORE splitting
    3. Performs stratified train-test split FIRST (zero index/hash overlap)
    4. Bounds only the training split if needed for hardware RAM limits
    5. Trains SVM model with stratified cross-validation inside a leakage-free Pipeline
    6. Saves artifacts
    """
    raw_df = preprocessing.load_raw_dataset()
    X_raw, y_raw = preprocessing.preprocess_dataframe(raw_df)
    X_clean, y_clean, audit_stats = preprocessing.clean_features_and_labels(
        X_raw, y_raw, drop_inf_nan=True, drop_duplicates=True
    )

    # Split FIRST on deduplicated, NaN/Inf-free data
    X_train_full, X_test, y_train_full, y_test, split_checks = preprocessing.stratified_clean_split(
        X_clean,
        y_clean,
        test_size=config.TEST_SPLIT_RATIO,
        random_state=config.RANDOM_STATE,
    )

    # Hardware cap applies strictly to the training split AFTER train/test separation
    if len(X_train_full) > sample_size:
        X_train, y_train = preprocessing.get_stratified_sample(
            X_train_full, y_train_full, max_samples=sample_size
        )
    else:
        X_train, y_train = X_train_full, y_train_full

    logger.info(
        f"Leakage-Free Dataset Split: Train={len(X_train):,} "
        f"(from pool {len(X_train_full):,}), Held-Out Test={len(X_test):,}"
    )

    pipeline, metadata = train_svm_pipeline(
        X_train, y_train, tune_hyperparameters=tune_grid, cv_folds=cv_folds
    )
    metadata["cleaning_audit"] = audit_stats
    metadata["split_audit"] = split_checks
    metadata["test_split_ratio"] = config.TEST_SPLIT_RATIO
    metadata["random_state"] = config.RANDOM_STATE
    save_model_artifacts(pipeline, metadata)

    return pipeline, metadata, X_test, y_test


if __name__ == "__main__":
    run_training_workflow(sample_size=config.SVM_TRAIN_SAMPLE_SIZE, tune_grid=True, cv_folds=5)
