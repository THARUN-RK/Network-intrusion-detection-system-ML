"""
Automated Unit and Integration Tests for IP Threat Intelligence and Forensic Auditing
"""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock

from backend.app.services.ip_validator import validate_and_classify_ip
from backend.app.services.threat_intel import (
    ThreatIntelCache,
    evaluate_hybrid_threat_status,
    analyze_ip_threat,
    THREAT_STATUS_NO_THREATS,
    THREAT_STATUS_SUSPICIOUS,
    THREAT_STATUS_MALICIOUS,
    THREAT_STATUS_UNKNOWN
)
import database
from backend.app.schemas import IPThreatLookupRequest
from backend.app.routes.threat_intel_routes import (
    check_ip_threat,
    get_threat_history,
    get_threat_kpis,
    export_ip_history_csv,
    export_flows_history_csv,
    get_dashboard_stats
)


@pytest.fixture(autouse=True)
def init_test_db():
    database.init_db()


def test_ip_validator_public_ipv4():
    res = validate_and_classify_ip("8.8.8.8")
    assert res["is_valid"] is True
    assert res["version"] == 4
    assert res["is_public"] is True
    assert res["is_private"] is False
    assert res["ip_type"] == "PUBLIC"


def test_ip_validator_public_ipv6():
    res = validate_and_classify_ip("2001:4860:4860::8888")
    assert res["is_valid"] is True
    assert res["version"] == 6
    assert res["is_public"] is True
    assert res["ip_type"] == "PUBLIC"


def test_ip_validator_private_rfc1918():
    for private_ip in ["192.168.1.1", "10.0.0.5", "172.16.50.1"]:
        res = validate_and_classify_ip(private_ip)
        assert res["is_valid"] is True
        assert res["is_public"] is False
        assert res["is_private"] is True
        assert res["ip_type"] == "PRIVATE"


def test_ip_validator_loopback_and_link_local():
    res_loop4 = validate_and_classify_ip("127.0.0.1")
    assert res_loop4["ip_type"] == "LOOPBACK"
    assert res_loop4["is_public"] is False

    res_loop6 = validate_and_classify_ip("::1")
    assert res_loop6["ip_type"] == "LOOPBACK"
    assert res_loop6["is_public"] is False

    res_link = validate_and_classify_ip("169.254.10.20")
    assert res_link["ip_type"] == "LINK_LOCAL"
    assert res_link["is_public"] is False


def test_ip_validator_invalid_inputs():
    for bad in ["not_an_ip", "999.999.999.999", "1.2.3", ""]:
        res = validate_and_classify_ip(bad)
        assert res["is_valid"] is False
        assert res["version"] is None


def test_threat_intel_cache():
    cache = ThreatIntelCache(ttl_seconds=2)
    cache.set("8.8.8.8", {"status": "clean"})
    
    cached = cache.get("8.8.8.8")
    assert cached is not None
    assert cached["status"] == "clean"

    # Missing entry
    assert cache.get("1.1.1.1") is None


def test_hybrid_risk_evaluation_malicious():
    val = validate_and_classify_ip("185.220.101.1")
    vt = {"available": True, "status": "SUCCESS", "malicious_count": 8, "suspicious_count": 1, "total_engines": 88}
    abuse = {"available": True, "status": "SUCCESS", "abuse_confidence_score": 90, "total_reports": 150, "is_tor": True}

    assessment = evaluate_hybrid_threat_status(val, vt, abuse)
    assert assessment["threat_status"] == THREAT_STATUS_MALICIOUS
    assert assessment["risk_score"] >= 50.0
    assert len(assessment["reasons"]) > 0


def test_hybrid_risk_evaluation_suspicious():
    val = validate_and_classify_ip("45.33.32.156")
    vt = {"available": True, "status": "SUCCESS", "malicious_count": 1, "suspicious_count": 2, "total_engines": 88}
    abuse = {"available": True, "status": "SUCCESS", "abuse_confidence_score": 30, "total_reports": 5, "is_tor": False}

    assessment = evaluate_hybrid_threat_status(val, vt, abuse)
    assert assessment["threat_status"] == THREAT_STATUS_SUSPICIOUS
    assert 20.0 <= assessment["risk_score"] < 50.0


def test_hybrid_risk_evaluation_clean():
    val = validate_and_classify_ip("8.8.8.8")
    vt = {"available": True, "status": "SUCCESS", "malicious_count": 0, "suspicious_count": 0, "total_engines": 88}
    abuse = {"available": True, "status": "SUCCESS", "abuse_confidence_score": 0, "total_reports": 0, "is_tor": False}

    assessment = evaluate_hybrid_threat_status(val, vt, abuse)
    assert assessment["threat_status"] == THREAT_STATUS_NO_THREATS
    assert assessment["risk_score"] == 0.0


def test_hybrid_risk_evaluation_private_ip():
    val = validate_and_classify_ip("192.168.1.100")
    assessment = evaluate_hybrid_threat_status(val, {}, {})
    assert assessment["threat_status"] == "UNKNOWN"
    assert "cannot be verified" in assessment["reasons"][0]


def test_route_check_ip_endpoint_private():
    req = IPThreatLookupRequest(ip_address="192.168.1.1", force_refresh=True)
    res = asyncio.run(check_ip_threat(req))
    assert res.ip_address == "192.168.1.1"
    assert res.validation["is_private"] is True
    assert res.threat_status == "UNKNOWN"


def test_route_check_ip_endpoint_public_mocked():
    with patch("backend.app.services.threat_intel.query_virustotal", new_callable=AsyncMock) as mock_vt, \
         patch("backend.app.services.threat_intel.query_abuseipdb", new_callable=AsyncMock) as mock_abuse:
        
        mock_vt.return_value = {
            "source": "VirusTotal v3",
            "status": "SUCCESS",
            "available": True,
            "malicious_count": 0,
            "suspicious_count": 0,
            "harmless_count": 70,
            "undetected_count": 10,
            "total_engines": 80,
            "detection_ratio": "0/80",
            "country": "US",
            "asn": "AS15169",
            "network": "8.8.8.0/24"
        }
        mock_abuse.return_value = {
            "source": "AbuseIPDB",
            "status": "SUCCESS",
            "available": True,
            "abuse_confidence_score": 0,
            "total_reports": 0,
            "distinct_users": 0,
            "country": "US",
            "isp": "Google LLC",
            "is_tor": False,
            "is_whitelisted": False
        }

        req = IPThreatLookupRequest(ip_address="8.8.8.8", force_refresh=True)
        res = asyncio.run(check_ip_threat(req))
        assert res.ip_address == "8.8.8.8"
        assert res.validation["is_public"] is True
        assert res.threat_status == "NO KNOWN THREATS"


def test_route_check_ip_invalid():
    req = IPThreatLookupRequest(ip_address="invalid.ip.format", force_refresh=True)
    res = asyncio.run(check_ip_threat(req))
    assert res.validation["is_valid"] is False
    assert res.threat_status == "UNKNOWN"


def test_route_history_and_kpis():
    # Perform a check first
    req = IPThreatLookupRequest(ip_address="10.0.0.1", force_refresh=True)
    asyncio.run(check_ip_threat(req))

    # Check history
    hist_data = get_threat_history(limit=10, offset=0)
    assert hist_data.total >= 1
    assert len(hist_data.items) >= 1

    # Check KPIs
    kpi_data = get_threat_kpis()
    assert kpi_data.total_lookups >= 1


def test_route_csv_exports():
    # IP CSV export
    res_ip_csv = export_ip_history_csv()
    assert res_ip_csv.status_code == 200
    assert "text/csv" in res_ip_csv.headers["content-type"]
    assert b"IP Address" in res_ip_csv.body

    # Flows CSV export
    res_flows_csv = export_flows_history_csv()
    assert res_flows_csv.status_code == 200
    assert "text/csv" in res_flows_csv.headers["content-type"]
    assert b"Predicted Category" in res_flows_csv.body


def test_route_dashboard_stats():
    stats = get_dashboard_stats()
    assert hasattr(stats, "total_records_analysed")
    assert hasattr(stats, "total_ips_checked")
    assert hasattr(stats, "threat_status_distribution")
