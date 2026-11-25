"""
Unit and Integration Tests for Model Evaluation & Day-Wise Temporal Generalization
"""

import pytest
import pandas as pd
from pathlib import Path

import config
import day_wise_evaluation
import evaluation


def test_partition_dataset_by_days():
    raw_csv = config.DATASET_RAW_DIR / "CICIDS2017_Benchmark_Sample.csv"
    if not raw_csv.exists():
        pytest.skip("Benchmark CSV not found")
        
    df = pd.read_csv(raw_csv, nrows=1000)
    splits = day_wise_evaluation.partition_dataset_by_days(df, random_state=42)
    
    assert set(splits.keys()) == {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
    for day, day_df in splits.items():
        assert len(day_df) > 0
        assert "Label" in day_df.columns


def test_day_wise_evaluation_execution():
    if not config.MODEL_PATH.exists():
        pytest.skip("Model not found")
        
    summary = day_wise_evaluation.evaluate_day_wise()
    assert "days" in summary
    assert len(summary["days"]) == 5
    assert summary["total_flows_evaluated"] > 0
    
    # Check that Friday has high detection rate
    assert summary["days"]["Friday"]["accuracy"] > 0.90
    assert summary["days"]["Friday"]["attack_detection_rate"] > 0.90
