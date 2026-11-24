"""
Unit tests for FastAPI endpoints and services.
"""

import pytest
from backend.app.main import health_check, get_model_information
from backend.app.routes.prediction import predict_single, predict_batch
from backend.app.routes.analytics import get_overview_analytics, get_attack_distribution
from backend.app.routes.history import get_history, get_alerts
from backend.app.schemas import FlowFeatures, BatchPredictionRequest
from backend.app.services.model_service import get_model_service


def test_health_check_function():
    result = health_check()
    assert result["status"] == "healthy"
    assert result["model_loaded"] is True
    assert result["version"] == "1.0.0"


def test_model_info_function():
    info = get_model_information()
    assert info["kernel"] == "rbf"
    assert len(info["classes"]) == 10
    assert "BENIGN" in info["classes"]
    assert "DoS" in info["classes"]


def test_predict_single_endpoint():
    flow = FlowFeatures(**{
        "Destination Port": 80,
        "Flow Duration": 15000.0,
        "Total Fwd Packets": 100,
        "Total Backward Packets": 2,
        "Flow Bytes/s": 2500000.0,
        "Flow Packets/s": 40000.0
    })
    service = get_model_service()
    res = predict_single(flow, service=service)
    assert res["prediction"] in service.predictor.classes
    assert res["status"] in ["BENIGN", "ATTACK"]
    assert 0.0 <= res["confidence"] <= 100.0


def test_predict_batch_endpoint():
    req = BatchPredictionRequest(records=[
        {"Destination Port": 80, "Flow Duration": 20000.0},
        {"Destination Port": 22, "Flow Duration": 40000.0}
    ])
    service = get_model_service()
    res = predict_batch(req, service=service)
    assert res["total_processed"] == 2
    assert len(res["results"]) == 2


def test_analytics_and_history_endpoints():
    overview = get_overview_analytics()
    assert "total_records_analysed" in overview
    assert "attack_detection_percentage" in overview

    attacks = get_attack_distribution()
    assert "category_distribution" in attacks

    history_res = get_history(limit=5, offset=0)
    assert "records" in history_res

    alerts_res = get_alerts(limit=5)
    assert "alerts" in alerts_res
