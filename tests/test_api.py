"""
Unit tests for FastAPI endpoints, services, authentication, and SOC management.
"""

import pytest
from backend.app.main import health_check, get_model_information, serve_soc_dashboard
from backend.app.routes.prediction import predict_single, predict_batch
from backend.app.routes.analytics import get_overview_analytics, get_attack_distribution
from backend.app.routes.history import get_history, get_alerts
from backend.app.routes.auth_routes import login, LoginRequest
from backend.app.routes.soc_routes import create_new_incident, list_incidents, get_ip_profiles, toggle_ip_blocking, generate_pdf_report
from backend.app.schemas import FlowFeatures, BatchPredictionRequest, IncidentCreateRequest, IPBlockRequest
from backend.app.services.model_service import get_model_service


def test_health_check_function():
    result = health_check()
    assert result["status"] == "healthy"
    assert result["model_loaded"] is True
    assert result["version"] in ["1.0.0", "2.0.0"]


def test_root_dashboard_endpoint():
    result = serve_soc_dashboard()
    assert result is not None
    # Verify FileResponse or HTMLResponse
    assert hasattr(result, "status_code") or hasattr(result, "path")


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
        "Flow Packets/s": 40000.0,
        "Source IP": "192.168.1.55",
        "Destination IP": "10.0.0.1"
    })
    service = get_model_service()
    res = predict_single(flow, service=service)
    assert res["prediction"] in service.predictor.classes or "ANOMALY" in res["prediction"]
    assert res["status"] in ["BENIGN", "ATTACK"]
    assert 0.0 <= res["confidence"] <= 100.0
    assert "severity" in res


def test_predict_batch_endpoint():
    req = BatchPredictionRequest(records=[
        {"Destination Port": 80, "Flow Duration": 20000.0, "Source IP": "192.168.1.10"},
        {"Destination Port": 22, "Flow Duration": 40000.0, "Source IP": "192.168.1.20"}
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


def test_auth_login():
    req = LoginRequest(username="admin", password="Admin@12345")
    token_resp = login(req)
    assert "access_token" in token_resp
    assert token_resp["role"] == "admin"
    assert token_resp["token_type"] == "bearer"


def test_soc_incident_workflow():
    req = IncidentCreateRequest(
        title="Automated Test Suspicious Flow",
        description="High rate DoS pattern detected from IP 192.168.1.200",
        severity="HIGH",
        source_ip="192.168.1.200",
        target_ip="10.0.0.1"
    )
    inc = create_new_incident(req)
    assert "incident_code" in inc
    assert inc["severity"] == "HIGH"

    all_incs = list_incidents()
    assert len(all_incs) >= 1


def test_soc_ip_reputation_and_block():
    block_req = IPBlockRequest(ip_address="192.168.1.99", block=True)
    resp = toggle_ip_blocking(block_req)
    assert resp["is_blocked"] is True

    ips = get_ip_profiles()
    assert isinstance(ips, list)


def test_soc_pdf_report_generation():
    pdf_resp = generate_pdf_report()
    assert pdf_resp.media_type == "application/pdf"
