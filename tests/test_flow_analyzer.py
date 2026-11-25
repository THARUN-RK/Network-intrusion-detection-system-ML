"""
Test Suite: IP-Based Network Attack Analyzer
Verifies IP validation, authentic flow retrieval, 71-feature SVM classification,
and API endpoints for the IP Attack Analyzer engine.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.ip_flow_analyzer import (
    analyze_ip_network_attacks,
    get_observed_ips_list,
    BENCHMARK_SAMPLE_IPS
)
from backend.app.services.ip_validator import validate_and_classify_ip


@pytest.fixture
def client():
    return TestClient(app)


def test_ip_validator_ipv4_and_ipv6():
    """Validates public and private IPv4 and IPv6 addresses."""
    val_v4 = validate_and_classify_ip("185.220.101.5")
    assert val_v4["is_valid"] is True
    assert val_v4["version"] == 4
    assert val_v4["is_public"] is True

    val_v6 = validate_and_classify_ip("2001:4860:4860::8888")
    assert val_v6["is_valid"] is True
    assert val_v6["version"] == 6
    assert val_v6["is_public"] is True

    val_private = validate_and_classify_ip("192.168.1.1")
    assert val_private["is_valid"] is True
    assert val_private["is_public"] is False

    val_invalid = validate_and_classify_ip("999.999.999.999")
    assert val_invalid["is_valid"] is False


def test_flow_analyzer_ddos_ip():
    """Tests authentic flow retrieval and SVM inference for known DDoS benchmark IP."""
    result = analyze_ip_network_attacks("185.220.101.5")

    assert result["has_data"] is True
    assert result["status"] == "COMPLETED"
    assert result["source_ip"] == "185.220.101.5"
    assert result["total_flows"] > 0
    assert result["malicious_flow_count"] > 0
    assert result["ddos_flow_count"] > 0
    assert "DDoS" in result["most_frequent_attack"]
    assert result["attack_severity"] in ["CRITICAL", "HIGH"]
    assert result["average_confidence"] > 50.0
    assert len(result["flow_records"]) > 0

    first_flow = result["flow_records"][0]
    assert "predicted_class" in first_flow
    assert first_flow["confidence"] > 0
    assert "distinction_note" in result
    assert "71 statistical flow features" in result["distinction_note"]


def test_flow_analyzer_portscan_ip():
    """Tests flow retrieval and SVM inference for known PortScan benchmark IP."""
    result = analyze_ip_network_attacks("45.33.32.156")

    assert result["has_data"] is True
    assert result["status"] == "COMPLETED"
    assert result["portscan_flow_count"] > 0
    assert result["malicious_flow_count"] > 0
    assert result["attack_severity"] in ["MEDIUM", "HIGH", "CRITICAL"]


def test_flow_analyzer_brute_force_ip():
    """Tests flow retrieval and SVM inference for known Brute Force (Patator) benchmark IP."""
    result = analyze_ip_network_attacks("194.26.29.112")

    assert result["has_data"] is True
    assert result["status"] == "COMPLETED"
    assert result["brute_force_flow_count"] > 0 or result["malicious_flow_count"] > 0


def test_flow_analyzer_benign_ip():
    """Tests flow retrieval and SVM inference for known Benign benchmark IP."""
    result = analyze_ip_network_attacks("192.168.1.101")

    assert result["has_data"] is True
    assert result["status"] == "COMPLETED"
    assert result["benign_flow_count"] > 0
    assert result["malicious_flow_count"] == 0
    assert result["attack_severity"] == "SAFE"


def test_flow_analyzer_insufficient_data():
    """Tests unobserved IP returning INSUFFICIENT DATA without guessing or faking flows."""
    result = analyze_ip_network_attacks("8.8.8.8")

    assert result["has_data"] is False
    assert result["status"] == "INSUFFICIENT DATA"
    assert result["total_flows"] == 0
    assert "No network traffic records found" in result["message"]
    assert "distinction_note" in result


def test_flow_analyzer_invalid_ip():
    """Tests malformed IP string returning INVALID_IP status."""
    result = analyze_ip_network_attacks("not_an_ip")

    assert result["has_data"] is False
    assert result["status"] == "INVALID_IP"
    assert result["total_flows"] == 0


def test_api_analyzer_endpoint(client):
    """Tests POST /api/intel/analyzer/ip endpoint."""
    response = client.post("/api/intel/analyzer/ip", json={"ip_address": "185.220.101.5"})
    assert response.status_code == 200
    data = response.json()

    assert data["has_data"] is True
    assert data["source_ip"] == "185.220.101.5"
    assert data["total_flows"] > 0
    assert data["ddos_flow_count"] > 0
    assert len(data["flow_records"]) > 0


def test_api_analyzer_alias_endpoint(client):
    """Tests POST /api/analyzer/ip alias endpoint."""
    response = client.post("/api/analyzer/ip", json={"ip_address": "8.8.8.8"})
    assert response.status_code == 200
    data = response.json()

    assert data["has_data"] is False
    assert data["status"] == "INSUFFICIENT DATA"
    assert data["total_flows"] == 0


def test_api_observed_ips_endpoint(client):
    """Tests GET /api/intel/analyzer/observed-ips endpoint."""
    response = client.get("/api/intel/analyzer/observed-ips")
    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) >= len(BENCHMARK_SAMPLE_IPS)
    ips = [item["ip_address"] for item in data]
    assert "185.220.101.5" in ips
    assert "192.168.1.101" in ips
