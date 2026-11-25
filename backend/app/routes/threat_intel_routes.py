"""
FastAPI Routes: IP Address Threat Intelligence & Unified Cybersecurity Operations
Provides endpoints for:
- Live IP threat intelligence checks (VirusTotal v3 + AbuseIPDB with caching and risk evaluation)
- IP threat lookup audit history (paginated, searchable, filterable)
- Threat intelligence KPI summaries
- One-click CSV export for IP threat history and network flow classification history
- Unified dashboard statistics combining ML flow classification with external threat intelligence
"""

import logging
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Query, Response, status

import database
from backend.app.schemas import (
    IPThreatLookupRequest,
    IPThreatLookupResponse,
    IPThreatHistoryResponse,
    IPThreatKpiResponse,
    UnifiedDashboardStatsResponse,
    IPAttackAnalyzerRequest,
    IPAttackAnalyzerResponse
)
from backend.app.services.threat_intel import analyze_ip_threat
from backend.app.services.ip_flow_analyzer import (
    analyze_ip_network_attacks,
    get_observed_ips_list
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/intel", tags=["Threat Intelligence & IP Reputation"])
analyzer_router = APIRouter(prefix="/api/analyzer", tags=["IP Attack Analyzer"])


@router.post(
    "/ip/check",
    response_model=IPThreatLookupResponse,
    summary="Check IP Address Threat Intelligence",
    description="Validates IPv4/IPv6, queries VirusTotal v3 and AbuseIPDB, assesses risk, and logs to audit trail."
)
async def check_ip_threat(payload: IPThreatLookupRequest):
    """
    Analyzes an IP address against threat intelligence engines.
    Saves the lookup result in the database for auditing and reporting.
    """
    raw_ip = payload.ip_address.strip()
    if not raw_ip:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="IP address cannot be empty."
        )

    try:
        report = await analyze_ip_threat(raw_ip, force_refresh=payload.force_refresh)

        # Extract values for database logging and client response
        vt = report.get("virustotal") or {}
        abuse = report.get("abuseipdb") or {}
        ip_version = report.get("ip_version") or 4
        is_public = bool(report.get("is_public", False))
        threat_status = report.get("threat_status", "UNKNOWN")
        risk_score = float(report.get("risk_score", 0.0))
        abuse_conf = float(abuse.get("abuse_confidence_score", 0.0))
        vt_ratio = str(vt.get("detection_ratio", "0/0"))
        total_reports = int(abuse.get("total_reports", 0))
        country = str(report.get("country", "Unknown"))
        isp = str(report.get("isp", "Unknown"))
        asn = str(report.get("asn", "Unknown"))
        reasons = report.get("reasons", [])

        val = {
            "version": ip_version,
            "ip_type": report.get("ip_type", "PUBLIC"),
            "is_public": is_public,
            "is_private": bool(report.get("is_private", not is_public)),
            "is_valid": report.get("success", True),
            "classification": f"IPv{ip_version} {report.get('ip_type', 'PUBLIC')}"
        }

        hybrid = {
            "status": threat_status,
            "risk_score": risk_score,
            "risk_level": report.get("risk_level", "INFO"),
            "reasons": reasons,
            "recommended_action": report.get("assessment_summary", "No action required.")
        }

        # Persist to database
        database.save_ip_threat_lookup(
            ip_address=report.get("ip_address", raw_ip),
            ip_version=ip_version,
            is_public=is_public,
            threat_status=threat_status,
            risk_score=risk_score,
            abuse_confidence=abuse_conf,
            vt_detection_ratio=vt_ratio,
            total_reports=total_reports,
            country=country,
            isp=isp,
            asn=asn,
            threat_reasons=reasons,
            raw_intel={
                "virustotal": vt,
                "abuseipdb": abuse,
                "hybrid": hybrid
            }
        )

        return IPThreatLookupResponse(
            ip_address=report.get("ip_address", raw_ip),
            validation=val,
            virustotal=vt,
            abuseipdb=abuse,
            hybrid_assessment=hybrid,
            threat_status=threat_status,
            risk_score=risk_score,
            cached=report.get("cached", False),
            checked_at=report.get("lookup_timestamp", "")
        )
    except Exception as e:
        logger.error(f"IP threat check failed for '{raw_ip}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Threat intelligence lookup encountered an error: {str(e)}"
        )


@router.get(
    "/ip/history",
    response_model=IPThreatHistoryResponse,
    summary="Get IP Threat Intelligence Lookup History",
    description="Returns paginated, searchable, and filterable history of previous IP threat lookups."
)
def get_threat_history(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    search: Optional[str] = None,
    threat_status: Optional[str] = None
):
    try:
        limit_val = limit if isinstance(limit, int) else 50
        offset_val = offset if isinstance(offset, int) else 0
        search_val = search if isinstance(search, str) else None
        threat_val = threat_status if isinstance(threat_status, str) else None

        data = database.get_ip_threat_history(
            limit=limit_val,
            offset=offset_val,
            search=search_val,
            threat_status=threat_val
        )
        return IPThreatHistoryResponse(total=data["total"], items=data["items"])
    except Exception as e:
        logger.error(f"Error fetching IP threat history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/ip/kpis",
    response_model=IPThreatKpiResponse,
    summary="Get IP Threat Intelligence KPI Summary",
    description="Returns aggregate KPI metrics for threat intelligence queries."
)
def get_threat_kpis():
    try:
        kpis = database.get_ip_threat_kpis()
        return IPThreatKpiResponse(**kpis)
    except Exception as e:
        logger.error(f"Error fetching threat KPIs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/ip/export/csv",
    summary="Export IP Threat History to CSV",
    description="Downloads the latest IP threat lookups as a downloadable CSV file."
)
def export_ip_history_csv():
    try:
        csv_content = database.export_ip_threat_history_csv()
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=ip_threat_history.csv",
                "Cache-Control": "no-cache"
            }
        )
    except Exception as e:
        logger.error(f"Error exporting IP threat CSV: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/flows/export/csv",
    summary="Export Network Flow Predictions to CSV",
    description="Downloads the latest classified network flows as a downloadable CSV file."
)
def export_flows_history_csv():
    try:
        csv_content = database.export_flow_history_csv()
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=network_flows_history.csv",
                "Cache-Control": "no-cache"
            }
        )
    except Exception as e:
        logger.error(f"Error exporting network flows CSV: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/dashboard-stats",
    response_model=UnifiedDashboardStatsResponse,
    summary="Get Unified Cybersecurity Dashboard Statistics",
    description="Aggregates both network flow ML classifications and IP threat intelligence metrics."
)
def get_dashboard_stats():
    try:
        overview = database.get_overview_statistics()
        threat_kpis = database.get_ip_threat_kpis()

        threat_status_distribution = {
            "NO KNOWN THREATS": threat_kpis.get("clean_count", 0),
            "SUSPICIOUS": threat_kpis.get("suspicious_count", 0),
            "MALICIOUS": threat_kpis.get("malicious_count", 0),
            "UNKNOWN": threat_kpis.get("unknown_count", 0),
        }

        return UnifiedDashboardStatsResponse(
            total_records_analysed=overview.get("total_records_analysed", 0),
            total_normal_connections=overview.get("total_normal_connections", 0),
            total_detected_attacks=overview.get("total_detected_attacks", 0),
            attack_detection_percentage=overview.get("attack_detection_percentage", 0.0),
            total_ips_checked=threat_kpis.get("total_lookups", 0),
            malicious_ips=threat_kpis.get("malicious_count", 0),
            suspicious_ips=threat_kpis.get("suspicious_count", 0),
            clean_ips=threat_kpis.get("clean_count", 0),
            category_distribution=overview.get("category_distribution", {}),
            threat_status_distribution=threat_status_distribution
        )
    except Exception as e:
        logger.error(f"Error generating dashboard stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# IP-BASED NETWORK ATTACK ANALYZER ENDPOINTS
# ==============================================================================

@router.post(
    "/analyzer/ip",
    response_model=IPAttackAnalyzerResponse,
    summary="Analyze Network Attacks by Source IP",
    description="Searches observed network traffic records for the entered IP, extracts 71 features, passes through the trained SVM model, and summarizes attack metrics."
)
@analyzer_router.post(
    "/ip",
    response_model=IPAttackAnalyzerResponse,
    summary="Analyze Network Attacks by Source IP",
    description="Searches observed network traffic records for the entered IP, extracts 71 features, passes through the trained SVM model, and summarizes attack metrics."
)
def analyze_ip_attacks_endpoint(payload: IPAttackAnalyzerRequest):
    """
    Evaluates empirical network session traffic for a specific source IP.
    Returns INSUFFICIENT DATA if no matching flows exist.
    """
    raw_ip = payload.ip_address.strip() if payload.ip_address else ""
    if not raw_ip:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Source IP address cannot be empty."
        )

    try:
        result = analyze_ip_network_attacks(raw_ip)
        return IPAttackAnalyzerResponse(**result)
    except Exception as e:
        logger.error(f"Error analyzing IP network attacks for '{raw_ip}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"IP attack analysis encountered an internal error: {str(e)}"
        )


@router.get(
    "/analyzer/observed-ips",
    summary="Get List of Observed Benchmark and Live Traffic Source IPs",
    description="Returns source IPs available for immediate empirical network attack analysis."
)
@analyzer_router.get(
    "/observed-ips",
    summary="Get List of Observed Benchmark and Live Traffic Source IPs",
    description="Returns source IPs available for immediate empirical network attack analysis."
)
def get_observed_ips_endpoint():
    """Returns sample observed source IPs with their known attack classifications."""
    try:
        return get_observed_ips_list()
    except Exception as e:
        logger.error(f"Error fetching observed IPs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
