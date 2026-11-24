"""
Unit tests for SVM prediction engine.
"""

import pytest
import numpy as np
import pandas as pd

import config
import predict
import database


@pytest.fixture(scope="module")
def predictor():
    return predict.get_predictor()


def test_predictor_loaded(predictor):
    assert predictor.pipeline is not None
    assert len(predictor.classes) == 10
    assert len(predictor.expected_features) == 71


def test_predict_single_flow(predictor):
    sample_flow = {
        "Destination Port": 80,
        "Flow Duration": 12000.0,
        "Total Fwd Packets": 150,
        "Total Backward Packets": 1,
        "Flow Bytes/s": 4000000.0,
        "Flow Packets/s": 50000.0,
        "SYN Flag Count": 1,
        "ACK Flag Count": 0
    }
    result = predictor.predict_single_flow(sample_flow, log_to_db=False)
    assert "prediction" in result
    assert "status" in result
    assert "confidence" in result
    assert result["status"] in ["BENIGN", "ATTACK"]
    assert 0.0 <= result["confidence"] <= 100.0


def test_predict_batch(predictor):
    df_sample = pd.DataFrame([
        {"Destination Port": 80, "Flow Duration": 500000.0},
        {"Destination Port": 21, "Flow Duration": 25000.0}
    ])
    results = predictor.predict_batch(df_sample, log_to_db=False)
    assert len(results) == 2
    assert "prediction" in results[0]
    assert "confidence" in results[0]
