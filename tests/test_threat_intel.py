"""
Automated Unit and Integration Tests for IP Threat Intelligence and Forensic Auditing.
Covers all configuration modes, HTTP error/rate-limit handling, independent API execution,
hybrid classification accuracy, and API endpoints using httpx.MockTransport (no fake data).
"""

import asyncio
import httpx
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

import config
import database
from backend.app.main import app
from backend.app.services.ip_validator import validate_and_classify_ip
from backend.app.services.threat_intel import (
    ThreatIntelCache,
    threat_intel_cache,
    evaluate_hybrid_threat_status,
    analyze_ip_threat,
    get_threat_intel_status,
    THREAT_STATUS_NO_THREATS,
    THREAT_STATUS_SUSPICIOUS,
    THREAT_STATUS_MALICIOUS,
    THREAT_STATUS_UNKNOWN
)
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
    threat_intel_cache.clear()


def _vt_ok_payload(malicious=0, suspicious=0, harmless=70, undetected=15, country="US", asn=15169, network="8.8.8.0/24"):
    return {
        "data": {
            "attributes": {
                "last_analysis_stats": {
                    "malicious": malicious,
                    "suspicious": suspicious,
                    "harmless": harmless,
                    "undetected": undetected
                },
                "country": country,
                "asn": asn,
                "as_owner": "GOOGLE",
                "network": network,
                "last_analysis_date": 1732550400,
                "reputation": 100 if malicious == 0 else -50
            }
        }
    }


def _abuse_ok_payload(score=0, reports=0, distinct=0, country="US", is_tor=False):
    return {
        "data": {
            "abuseConfidenceScore": score,
            "totalReports": reports,
            "numDistinctUsers": distinct,
            "countryCode": country,
            "countryName": "United States" if country == "US" else "Unknown",
            "isp": "Google LLC",
            "domain": "google.com",
            "usageType": "Data Center/Web Hosting/Transit",
            "isTor": is_tor,
            "isWhitelisted": (score == 0),
            "lastReportedAt": "2026-11-25T10:00:00+00:00" if reports > 0 else None
        }
    }


# ==============================================================================
# IP VALIDATOR & CACHE TESTS
# ==============================================================================

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
    for bad in ["not_an_ip", "999.999.999.999", "1.2.3", "", "8.8.8.8/24", "8.8.8.8:80"]:
        res = validate_and_classify_ip(bad)
        assert res["is_valid"] is False
        assert res["version"] is None


def test_threat_intel_cache():
    cache = ThreatIntelCache(ttl_seconds=2)
    cache.set("8.8.8.8", {"status": "clean"})
    cached = cache.get("8.8.8.8")
    assert cached is not None
    assert cached["status"] == "clean"
    assert cache.get("1.1.1.1") is None


# ==============================================================================
# ENV CONFIG & SECRET HYGIENE TESTS
# ==============================================================================

def test_env_loading_and_placeholder_filtering(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('VIRUSTOTAL_API_KEY="vt_secret_123"\nABUSEIPDB_API_KEY=your_api_key_here\n')

    monkeypatch.delenv("VIRUSTOTAL_API_KEY", raising=False)
    monkeypatch.delenv("ABUSEIPDB_API_KEY", raising=False)

    config.load_environment(env_file)
    assert config.get_api_key("VIRUSTOTAL_API_KEY") == "vt_secret_123"
    assert config.get_api_key("ABUSEIPDB_API_KEY") == ""

    # System / Render environment variable must always win over local .env
    monkeypatch.setenv("VIRUSTOTAL_API_KEY", "render_override_secret")
    config.load_environment(env_file)
    assert config.get_api_key("VIRUSTOTAL_API_KEY") == "render_override_secret"


def test_status_endpoint_never_exposes_secrets(monkeypatch):
    monkeypatch.setenv("VIRUSTOTAL_API_KEY", "super_secret_vt_value_do_not_leak")
    monkeypatch.delenv("ABUSEIPDB_API_KEY", raising=False)

    client = TestClient(app)
    resp = client.get("/api/intel/status")
    assert resp.status_code == 200
    body_text = resp.text
    assert "super_secret_vt_value_do_not_leak" not in body_text

    data = resp.json()
    assert data["virustotal"]["configured"] is True
    assert data["abuseipdb"]["configured"] is False
    assert data["abuseipdb"]["status"] == "NOT_CONFIGURED"


# ==============================================================================
# UNIT HYBRID RISK CLASSIFICATION TESTS
# ==============================================================================

def test_hybrid_risk_evaluation_malicious():
    val = validate_and_classify_ip("185.220.101.1")
    vt = {"available": True, "status": "SUCCESS", "malicious_count": 8, "suspicious_count": 1, "total_engines": 88}
    abuse = {"available": True, "status": "SUCCESS", "abuse_confidence_score": 90, "total_reports": 150, "is_tor": True}

    assessment = evaluate_hybrid_threat_status(val, vt, abuse)
    assert assessment["threat_status"] == THREAT_STATUS_MALICIOUS
    assert assessment["risk_score"] >= 50.0
    assert assessment["risk_score_available"] is True
    assert len(assessment["reasons"]) > 0


def test_hybrid_risk_evaluation_suspicious():
    val = validate_and_classify_ip("45.33.32.156")
    vt = {"available": True, "status": "SUCCESS", "malicious_count": 1, "suspicious_count": 2, "total_engines": 88}
    abuse = {"available": True, "status": "SUCCESS", "abuse_confidence_score": 30, "total_reports": 5, "is_tor": False}

    assessment = evaluate_hybrid_threat_status(val, vt, abuse)
    assert assessment["threat_status"] == THREAT_STATUS_SUSPICIOUS
    assert 20.0 <= assessment["risk_score"] < 50.0
    assert assessment["risk_score_available"] is True


def test_hybrid_risk_evaluation_clean():
    val = validate_and_classify_ip("8.8.8.8")
    vt = {"available": True, "status": "SUCCESS", "malicious_count": 0, "suspicious_count": 0, "total_engines": 88}
    abuse = {"available": True, "status": "SUCCESS", "abuse_confidence_score": 0, "total_reports": 0, "is_tor": False}

    assessment = evaluate_hybrid_threat_status(val, vt, abuse)
    assert assessment["threat_status"] == THREAT_STATUS_NO_THREATS
    assert assessment["classification_label"] == "BENIGN / NO KNOWN THREATS"
    assert assessment["risk_score"] == 0.0
    assert assessment["risk_score_available"] is True


def test_hybrid_risk_evaluation_private_ip():
    val = validate_and_classify_ip("192.168.1.100")
    assessment = evaluate_hybrid_threat_status(val, {}, {})
    assert assessment["threat_status"] == THREAT_STATUS_UNKNOWN
    assert assessment["risk_score"] is None
    assert assessment["risk_score_available"] is False
    assert "cannot be verified" in assessment["reasons"][0]


# ==============================================================================
# TASK 7: ALL 8 REQUIRED SCENARIOS (USING MOCK TRANSPORT)
# ==============================================================================

def test_scenario_1_both_apis_configured(monkeypatch):
    """Scenario 1: Both APIs configured and returning full telemetry."""
    monkeypatch.setenv("VIRUSTOTAL_API_KEY", "vt_key_valid")
    monkeypatch.setenv("ABUSEIPDB_API_KEY", "abuse_key_valid")

    def handler(request: httpx.Request) -> httpx.Response:
        if "virustotal.com" in str(request.url):
            assert request.headers.get("x-apikey") == "vt_key_valid"
            return httpx.Response(200, json=_vt_ok_payload(malicious=6, suspicious=2, harmless=60, undetected=20))
        if "abuseipdb.com" in str(request.url):
            assert request.headers.get("Key") == "abuse_key_valid"
            return httpx.Response(200, json=_abuse_ok_payload(score=88, reports=45, distinct=12, is_tor=True))
        return httpx.Response(404)

    async def _run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await analyze_ip_threat("185.220.101.1", force_refresh=True, http_client=client)

    res = asyncio.run(_run())
    assert res["threat_status"] == THREAT_STATUS_MALICIOUS
    assert res["risk_score_available"] is True
    assert res["risk_score"] == 88.0
    # Verify ALL VT fields preserved
    vt = res["virustotal"]
    assert vt["configured"] is True
    assert vt["status"] == "SUCCESS"
    assert vt["connection_status"] == "Connected"
    assert vt["malicious_count"] == 6
    assert vt["suspicious_count"] == 2
    assert vt["harmless_count"] == 60
    assert vt["undetected_count"] == 20
    assert vt["total_engines"] == 88
    assert vt["country"] == "US"
    assert vt["asn"] == "AS15169"
    assert vt["network"] == "8.8.8.0/24"
    assert "UTC" in vt["last_analysis_date"]
    # Verify ALL AbuseIPDB fields preserved
    abuse = res["abuseipdb"]
    assert abuse["configured"] is True
    assert abuse["status"] == "SUCCESS"
    assert abuse["abuse_confidence_score"] == 88
    assert abuse["total_reports"] == 45
    assert abuse["num_distinct_users"] == 12
    assert abuse["isp"] == "Google LLC"
    assert abuse["domain"] == "google.com"
    assert abuse["usage_type"] == "Data Center/Web Hosting/Transit"
    assert abuse["is_tor"] is True
    assert abuse["last_reported_at"] != "Never"


def test_scenario_2_only_virustotal_configured(monkeypatch):
    """Scenario 2: Only VirusTotal configured; AbuseIPDB unconfigured."""
    monkeypatch.setenv("VIRUSTOTAL_API_KEY", "vt_key_valid")
    monkeypatch.delenv("ABUSEIPDB_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        if "virustotal.com" in str(request.url):
            return httpx.Response(200, json=_vt_ok_payload(malicious=0, suspicious=0, harmless=72, undetected=10))
        raise AssertionError("AbuseIPDB should not be called without API key")

    async def _run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await analyze_ip_threat("8.8.8.8", force_refresh=True, http_client=client)

    res = asyncio.run(_run())
    assert res["threat_status"] == THREAT_STATUS_NO_THREATS
    assert res["classification_label"] == "BENIGN / NO KNOWN THREATS"
    assert res["virustotal"]["available"] is True
    assert res["virustotal"]["total_engines"] == 82
    assert res["abuseipdb"]["status"] == "NOT_CONFIGURED"
    assert res["abuseipdb"]["configured"] is False
    assert any("AbuseIPDB" in reason for reason in res["reasons"])


def test_scenario_2b_only_virustotal_404_not_falsely_safe(monkeypatch):
    """VT returning 404 (0 vendor engines) when AbuseIPDB is unavailable must NOT classify IP as safe."""
    monkeypatch.setenv("VIRUSTOTAL_API_KEY", "vt_key_valid")
    monkeypatch.delenv("ABUSEIPDB_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async def _run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await analyze_ip_threat("1.2.3.4", force_refresh=True, http_client=client)

    res = asyncio.run(_run())
    assert res["threat_status"] == THREAT_STATUS_UNKNOWN
    assert res["risk_score"] is None
    assert res["risk_score_available"] is False


def test_scenario_3_only_abuseipdb_configured(monkeypatch):
    """Scenario 3: Only AbuseIPDB configured; VirusTotal unconfigured."""
    monkeypatch.delenv("VIRUSTOTAL_API_KEY", raising=False)
    monkeypatch.setenv("ABUSEIPDB_API_KEY", "abuse_key_valid")

    def handler(request: httpx.Request) -> httpx.Response:
        if "abuseipdb.com" in str(request.url):
            return httpx.Response(200, json=_abuse_ok_payload(score=35, reports=4, distinct=3))
        raise AssertionError("VirusTotal should not be called without API key")

    async def _run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await analyze_ip_threat("45.33.32.156", force_refresh=True, http_client=client)

    res = asyncio.run(_run())
    assert res["threat_status"] == THREAT_STATUS_SUSPICIOUS
    assert res["risk_score"] == 35.0
    assert res["virustotal"]["status"] == "NOT_CONFIGURED"
    assert res["abuseipdb"]["available"] is True
    assert any("VirusTotal" in reason for reason in res["reasons"])


def test_scenario_4_neither_api_configured(monkeypatch):
    """Scenario 4: Neither API configured -> UNKNOWN, no misleading 0/100 score, never cached."""
    monkeypatch.delenv("VIRUSTOTAL_API_KEY", raising=False)
    monkeypatch.delenv("ABUSEIPDB_API_KEY", raising=False)

    res = asyncio.run(analyze_ip_threat("8.8.4.4", force_refresh=True))
    assert res["threat_status"] == THREAT_STATUS_UNKNOWN
    assert res["risk_score"] is None
    assert res["risk_score_available"] is False
    assert res["virustotal"]["status"] == "NOT_CONFIGURED"
    assert res["abuseipdb"]["status"] == "NOT_CONFIGURED"
    assert "not configured" in res["assessment_summary"].lower()
    # Must not be cached cleanly as actionable intelligence
    assert threat_intel_cache.get("8.8.4.4") is None


def test_scenario_5_invalid_api_key_401_and_403_fault_isolation(monkeypatch):
    """Scenario 5: Invalid/revoked API key (401 on VT, 403 on AbuseIPDB, and independent fallback)."""
    monkeypatch.setenv("VIRUSTOTAL_API_KEY", "bad_vt_key")
    monkeypatch.setenv("ABUSEIPDB_API_KEY", "good_abuse_key")

    def handler_partial(request: httpx.Request) -> httpx.Response:
        if "virustotal.com" in str(request.url):
            return httpx.Response(401, json={"error": {"message": "Wrong credentials"}})
        if "abuseipdb.com" in str(request.url):
            return httpx.Response(200, json=_abuse_ok_payload(score=95, reports=60, distinct=20))
        return httpx.Response(404)

    async def _run_partial():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler_partial)) as client:
            return await analyze_ip_threat("45.155.205.233", force_refresh=True, http_client=client)

    res = asyncio.run(_run_partial())
    assert res["virustotal"]["status"] == "UNAUTHORIZED"
    assert res["abuseipdb"]["status"] == "SUCCESS"
    assert res["threat_status"] == THREAT_STATUS_MALICIOUS
    assert res["risk_score"] == 95.0

    # Also test 403 Forbidden on both
    def handler_403(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    async def _run_403():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler_403)) as client:
            return await analyze_ip_threat("45.155.205.233", force_refresh=True, http_client=client)

    res_403 = asyncio.run(_run_403())
    assert res_403["virustotal"]["status"] == "FORBIDDEN"
    assert res_403["abuseipdb"]["status"] == "FORBIDDEN"
    assert res_403["threat_status"] == THREAT_STATUS_UNKNOWN
    assert res_403["risk_score"] is None


def test_scenario_6_invalid_ip_address():
    """Scenario 6: Invalid IP address handling."""
    res = asyncio.run(analyze_ip_threat("not_a_valid_ip", force_refresh=True))
    assert res["validation"]["is_valid"] is False
    assert res["threat_status"] == THREAT_STATUS_UNKNOWN
    assert res["risk_score"] is None
    assert res["risk_score_available"] is False


def test_scenario_7_private_ip_address(monkeypatch):
    """Scenario 7: Private IP addresses never call external APIs and return UNKNOWN with no fake risk score."""
    monkeypatch.setenv("VIRUSTOTAL_API_KEY", "vt_key")
    monkeypatch.setenv("ABUSEIPDB_API_KEY", "abuse_key")

    res = asyncio.run(analyze_ip_threat("192.168.1.50", force_refresh=True))
    assert res["is_public"] is False
    assert res["is_private"] is True
    assert res["threat_status"] == THREAT_STATUS_UNKNOWN
    assert res["risk_score"] is None
    assert res["risk_score_available"] is False
    assert res["virustotal"]["status"] == "NOT_APPLICABLE"
    assert res["abuseipdb"]["status"] == "NOT_APPLICABLE"


def test_scenario_8_api_rate_limit_timeout_5xx_and_null_fields(monkeypatch):
    """Scenario 8: HTTP 429 rate limits, network timeouts, 5xx errors, and missing JSON fields."""
    monkeypatch.setenv("VIRUSTOTAL_API_KEY", "vt_key")
    monkeypatch.setenv("ABUSEIPDB_API_KEY", "abuse_key")

    # 8a: Both rate-limited (429)
    def handler_429(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "60"})

    async def _run_429():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler_429)) as client:
            return await analyze_ip_threat("1.1.1.1", force_refresh=True, http_client=client)

    res_429 = asyncio.run(_run_429())
    assert res_429["virustotal"]["status"] == "RATE_LIMITED"
    assert res_429["abuseipdb"]["status"] == "RATE_LIMITED"
    assert "60s" in res_429["virustotal"]["error"]
    assert res_429["threat_status"] == THREAT_STATUS_UNKNOWN
    assert res_429["risk_score"] is None

    # 8b: Timeout on VT + Server 503 on AbuseIPDB
    def handler_timeout_503(request: httpx.Request) -> httpx.Response:
        if "virustotal.com" in str(request.url):
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(503)

    async def _run_t_503():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler_timeout_503)) as client:
            return await analyze_ip_threat("1.1.1.2", force_refresh=True, http_client=client)

    res_t503 = asyncio.run(_run_t_503())
    assert res_t503["virustotal"]["status"] == "TIMEOUT"
    assert res_t503["abuseipdb"]["status"] == "SERVER_ERROR"
    assert res_t503["threat_status"] == THREAT_STATUS_UNKNOWN
    assert res_t503["risk_score"] is None

    # 8c: Missing/null JSON fields on HTTP 200 must never raise TypeError/AttributeError
    def handler_null_fields(request: httpx.Request) -> httpx.Response:
        if "virustotal.com" in str(request.url):
            return httpx.Response(200, json={
                "data": {
                    "attributes": {
                        "last_analysis_stats": None,
                        "country": None,
                        "asn": None,
                        "last_analysis_results": {"engine1": {}}
                    }
                }
            })
        return httpx.Response(200, json={
            "data": {
                "abuseConfidenceScore": None,
                "totalReports": None,
                "isp": None,
                "lastReportedAt": None
            }
        })

    async def _run_null():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler_null_fields)) as client:
            return await analyze_ip_threat("1.1.1.3", force_refresh=True, http_client=client)

    res_null = asyncio.run(_run_null())
    assert res_null["virustotal"]["status"] == "SUCCESS"
    assert res_null["abuseipdb"]["status"] == "SUCCESS"
    assert res_null["threat_status"] == THREAT_STATUS_NO_THREATS


# ==============================================================================
# ROUTE INTEGRATION & AUDIT HISTORY TESTS
# ==============================================================================

def test_route_check_ip_endpoint_private():
    req = IPThreatLookupRequest(ip_address="192.168.1.1", force_refresh=True)
    res = asyncio.run(check_ip_threat(req))
    assert res.ip_address == "192.168.1.1"
    assert res.validation["is_private"] is True
    assert res.threat_status == "UNKNOWN"
    assert res.risk_score is None
    assert res.risk_score_available is False


def test_route_check_ip_endpoint_public_mocked():
    with patch("backend.app.services.threat_intel.query_virustotal", new_callable=AsyncMock) as mock_vt, \
         patch("backend.app.services.threat_intel.query_abuseipdb", new_callable=AsyncMock) as mock_abuse:

        mock_vt.return_value = {
            "source": "VirusTotal v3",
            "configured": True,
            "status": "SUCCESS",
            "connection_status": "Connected",
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
            "configured": True,
            "status": "SUCCESS",
            "connection_status": "Connected",
            "available": True,
            "abuse_confidence_score": 0,
            "total_reports": 0,
            "num_distinct_users": 0,
            "country": "US",
            "isp": "Google LLC",
            "domain": "google.com",
            "usage_type": "Data Center/Web Hosting/Transit",
            "is_tor": False,
            "is_whitelisted": True
        }

        req = IPThreatLookupRequest(ip_address="8.8.8.8", force_refresh=True)
        res = asyncio.run(check_ip_threat(req))
        assert res.ip_address == "8.8.8.8"
        assert res.validation["is_public"] is True
        assert res.threat_status == "NO KNOWN THREATS"
        assert res.hybrid_assessment["classification_label"] == "BENIGN / NO KNOWN THREATS"
        assert res.risk_score == 0.0
        assert res.risk_score_available is True
        assert res.virustotal["harmless_count"] == 70
        assert res.virustotal["network"] == "8.8.8.0/24"
        assert res.abuseipdb["domain"] == "google.com"


def test_route_check_ip_invalid():
    req = IPThreatLookupRequest(ip_address="invalid.ip.format", force_refresh=True)
    res = asyncio.run(check_ip_threat(req))
    assert res.validation["is_valid"] is False
    assert res.threat_status == "UNKNOWN"
    assert res.risk_score is None
    assert res.risk_score_available is False


def test_route_history_and_kpis():
    req = IPThreatLookupRequest(ip_address="10.0.0.1", force_refresh=True)
    asyncio.run(check_ip_threat(req))

    hist_data = get_threat_history(limit=10, offset=0)
    assert hist_data.total >= 1
    assert len(hist_data.items) >= 1

    kpi_data = get_threat_kpis()
    assert kpi_data.total_lookups >= 1


def test_route_csv_exports():
    res_ip_csv = export_ip_history_csv()
    assert res_ip_csv.status_code == 200
    assert "text/csv" in res_ip_csv.headers["content-type"]
    assert b"IP Address" in res_ip_csv.body

    res_flows_csv = export_flows_history_csv()
    assert res_flows_csv.status_code == 200
    assert "text/csv" in res_flows_csv.headers["content-type"]
    assert b"Predicted Category" in res_flows_csv.body


def test_route_dashboard_stats():
    stats = get_dashboard_stats()
    assert hasattr(stats, "total_records_analysed")
    assert hasattr(stats, "total_ips_checked")
    assert hasattr(stats, "threat_status_distribution")
