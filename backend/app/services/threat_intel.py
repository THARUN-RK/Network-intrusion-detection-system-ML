"""
Hybrid IP Threat Intelligence Service
Integrates VirusTotal API v3 and AbuseIPDB API with caching, rate-limit resilience,
configurable risk assessment rules, and strict separation from ML flow classification.
"""

import os
import time
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

import httpx

from backend.app.services.ip_validator import validate_and_classify_ip

logger = logging.getLogger(__name__)

# Threat Intelligence API Configuration
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY", "").strip()
ABUSEIPDB_API_KEY = os.getenv("ABUSEIPDB_API_KEY", "").strip()

VT_API_URL = "https://www.virustotal.com/api/v3/ip_addresses"
ABUSEIPDB_API_URL = "https://api.abuseipdb.com/api/v2/check"

# Cache TTL: 24 hours (86,400 seconds)
CACHE_TTL_SECONDS = int(os.getenv("THREAT_INTEL_CACHE_TTL", "86400"))

# Canonical Threat Statuses
THREAT_STATUS_NO_THREATS = "NO KNOWN THREATS"
THREAT_STATUS_SUSPICIOUS = "SUSPICIOUS"
THREAT_STATUS_MALICIOUS = "MALICIOUS"
THREAT_STATUS_UNKNOWN = "UNKNOWN"


class ThreatIntelCache:
    """Thread-safe in-memory cache with expiration TTL to respect API quotas."""
    def __init__(self, ttl_seconds: int = CACHE_TTL_SECONDS):
        self.ttl = ttl_seconds
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def get(self, ip: str) -> Optional[Dict[str, Any]]:
        if ip in self._cache:
            timestamp, data = self._cache[ip]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self._cache[ip]
        return None

    def set(self, ip: str, data: Dict[str, Any]) -> None:
        self._cache[ip] = (time.time(), data)

    def clear(self) -> None:
        self._cache.clear()


# Global cache instance
threat_intel_cache = ThreatIntelCache()


async def query_virustotal(ip: str, client: httpx.AsyncClient) -> Dict[str, Any]:
    """
    Queries VirusTotal API v3 for IP reputation.
    Returns structured results or graceful error states.
    """
    api_key = os.getenv("VIRUSTOTAL_API_KEY", VIRUSTOTAL_API_KEY).strip()
    if not api_key:
        return {
            "source": "VirusTotal v3",
            "status": "NOT_CONFIGURED",
            "available": False,
            "error": "VIRUSTOTAL_API_KEY environment variable is not configured."
        }

    url = f"{VT_API_URL}/{ip}"
    headers = {
        "x-apikey": api_key,
        "Accept": "application/json"
    }

    try:
        response = await client.get(url, headers=headers, timeout=8.0)

        if response.status_code == 200:
            payload = response.json().get("data", {}).get("attributes", {})
            stats = payload.get("last_analysis_stats", {})
            
            malicious = stats.get("malicious", 0)
            suspicious = stats.get("suspicious", 0)
            harmless = stats.get("harmless", 0)
            undetected = stats.get("undetected", 0)
            total_engines = sum([malicious, suspicious, harmless, undetected])

            last_epoch = payload.get("last_analysis_date")
            last_date_str = (
                datetime.fromtimestamp(last_epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                if last_epoch else "N/A"
            )

            return {
                "source": "VirusTotal v3",
                "status": "SUCCESS",
                "available": True,
                "malicious_count": malicious,
                "suspicious_count": suspicious,
                "harmless_count": harmless,
                "undetected_count": undetected,
                "total_engines": total_engines,
                "detection_ratio": f"{malicious}/{total_engines}" if total_engines > 0 else "0/0",
                "country": payload.get("country", "Unknown"),
                "asn": payload.get("asn", 0),
                "as_owner": payload.get("as_owner", "Unknown"),
                "network": payload.get("network", "Unknown"),
                "last_analysis_date": last_date_str,
                "reputation_raw": payload.get("reputation", 0)
            }
        elif response.status_code == 404:
            return {
                "source": "VirusTotal v3",
                "status": "NO_RECORD",
                "available": True,
                "malicious_count": 0,
                "suspicious_count": 0,
                "harmless_count": 0,
                "undetected_count": 0,
                "total_engines": 0,
                "detection_ratio": "0/0",
                "country": "Unknown",
                "asn": 0,
                "as_owner": "Unknown",
                "network": "Unknown",
                "last_analysis_date": "N/A",
                "message": "IP address not currently indexed in VirusTotal database."
            }
        elif response.status_code == 429:
            logger.warning(f"[VirusTotal] Rate limit exceeded querying {ip}")
            return {
                "source": "VirusTotal v3",
                "status": "RATE_LIMITED",
                "available": False,
                "error": "VirusTotal API rate limit reached (4 requests/min on public tier)."
            }
        elif response.status_code in [401, 403]:
            return {
                "source": "VirusTotal v3",
                "status": "UNAUTHORIZED",
                "available": False,
                "error": "Invalid or revoked VirusTotal API key."
            }
        else:
            return {
                "source": "VirusTotal v3",
                "status": "ERROR",
                "available": False,
                "error": f"VirusTotal returned HTTP {response.status_code}"
            }
    except httpx.TimeoutException:
        logger.warning(f"[VirusTotal] Request timed out for {ip}")
        return {
            "source": "VirusTotal v3",
            "status": "TIMEOUT",
            "available": False,
            "error": "VirusTotal API request timed out (limit: 8.0s)."
        }
    except Exception as e:
        logger.error(f"[VirusTotal] Connection failure for {ip}: {e}")
        return {
            "source": "VirusTotal v3",
            "status": "CONNECTION_ERROR",
            "available": False,
            "error": str(e)
        }


async def query_abuseipdb(ip: str, client: httpx.AsyncClient) -> Dict[str, Any]:
    """
    Queries AbuseIPDB API v2 for IP abuse reports and confidence score.
    Returns structured results or graceful error states.
    """
    api_key = os.getenv("ABUSEIPDB_API_KEY", ABUSEIPDB_API_KEY).strip()
    if not api_key:
        return {
            "source": "AbuseIPDB",
            "status": "NOT_CONFIGURED",
            "available": False,
            "error": "ABUSEIPDB_API_KEY environment variable is not configured."
        }

    headers = {
        "Key": api_key,
        "Accept": "application/json"
    }
    params = {
        "ipAddress": ip,
        "maxAgeInDays": "90",
        "verbose": "true"
    }

    try:
        response = await client.get(ABUSEIPDB_API_URL, headers=headers, params=params, timeout=8.0)

        if response.status_code == 200:
            payload = response.json().get("data", {})
            return {
                "source": "AbuseIPDB",
                "status": "SUCCESS",
                "available": True,
                "abuse_confidence_score": payload.get("abuseConfidenceScore", 0),
                "total_reports": payload.get("totalReports", 0),
                "num_distinct_users": payload.get("numDistinctUsers", 0),
                "country": payload.get("countryCode", "Unknown"),
                "country_name": payload.get("countryName", "Unknown"),
                "isp": payload.get("isp", "Unknown"),
                "domain": payload.get("domain", "Unknown"),
                "usage_type": payload.get("usageType", "Unknown"),
                "is_tor": bool(payload.get("isTor", False)),
                "is_whitelisted": bool(payload.get("isWhitelisted", False)),
                "last_reported_at": payload.get("lastReportedAt", "Never")
            }
        elif response.status_code == 429:
            logger.warning(f"[AbuseIPDB] Rate limit exceeded querying {ip}")
            return {
                "source": "AbuseIPDB",
                "status": "RATE_LIMITED",
                "available": False,
                "error": "AbuseIPDB daily API rate limit reached (1,000 queries/day on free tier)."
            }
        elif response.status_code in [401, 403]:
            return {
                "source": "AbuseIPDB",
                "status": "UNAUTHORIZED",
                "available": False,
                "error": "Invalid or revoked AbuseIPDB API key."
            }
        else:
            return {
                "source": "AbuseIPDB",
                "status": "ERROR",
                "available": False,
                "error": f"AbuseIPDB returned HTTP {response.status_code}"
            }
    except httpx.TimeoutException:
        logger.warning(f"[AbuseIPDB] Request timed out for {ip}")
        return {
            "source": "AbuseIPDB",
            "status": "TIMEOUT",
            "available": False,
            "error": "AbuseIPDB API request timed out (limit: 8.0s)."
        }
    except Exception as e:
        logger.error(f"[AbuseIPDB] Connection failure for {ip}: {e}")
        return {
            "source": "AbuseIPDB",
            "status": "CONNECTION_ERROR",
            "available": False,
            "error": str(e)
        }


def evaluate_hybrid_threat_status(
    ip_meta: Dict[str, Any],
    vt_res: Dict[str, Any],
    abuse_res: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Configurable Risk Assessment Engine.
    
    Produces one of 4 canonical statuses:
    1. NO KNOWN THREATS
    2. SUSPICIOUS
    3. MALICIOUS
    4. UNKNOWN
    
    Principles:
    - Distinguishes 'No known reports' from 'Insufficient info' and 'Confirmed malicious'.
    - Tor exit node status alone does NOT establish malicious activity.
    - Never evaluates an IP based on its numerical value alone.
    """
    reasons = []
    
    # 1. Non-public IP addresses cannot be established as public threats
    if not ip_meta.get("is_public", False):
        return {
            "threat_status": "UNKNOWN",
            "risk_level": "INFO",
            "risk_score": 0.0,
            "is_malicious": False,
            "is_suspicious": False,
            "reasons": [
                f"{ip_meta.get('ip_type')} range cannot be verified against public threat feeds."
            ],
            "assessment_summary": f"Internal or non-routable IP address ({ip_meta.get('ip_type')}). No external threat intelligence applies."
        }

    # 2. Check API availability
    vt_available = vt_res.get("available", False) and vt_res.get("status") in ["SUCCESS", "NO_RECORD"]
    abuse_available = abuse_res.get("available", False) and abuse_res.get("status") == "SUCCESS"

    if not vt_available and not abuse_available:
        err_details = []
        if vt_res.get("error"):
            err_details.append(f"VirusTotal: {vt_res.get('error')}")
        if abuse_res.get("error"):
            err_details.append(f"AbuseIPDB: {abuse_res.get('error')}")
        return {
            "threat_status": "UNKNOWN",
            "risk_level": "UNKNOWN",
            "risk_score": 0.0,
            "is_malicious": False,
            "is_suspicious": False,
            "reasons": ["All threat intelligence feeds are currently unavailable or unconfigured."] + err_details,
            "assessment_summary": "Insufficient information: Threat intelligence APIs did not return actionable intelligence."
        }

    # Extract metrics
    vt_malicious = vt_res.get("malicious_count", 0) if vt_available else 0
    vt_suspicious = vt_res.get("suspicious_count", 0) if vt_available else 0
    vt_total = vt_res.get("total_engines", 0) if vt_available else 0

    abuse_score = abuse_res.get("abuse_confidence_score", 0) if abuse_available else 0
    abuse_reports = abuse_res.get("total_reports", 0) if abuse_available else 0
    is_tor = abuse_res.get("is_tor", False) if abuse_available else False
    is_whitelisted = abuse_res.get("is_whitelisted", False) if abuse_available else False

    # Normalized risk score calculation (0 - 100)
    vt_ratio_score = (vt_malicious / max(1, vt_total)) * 100.0 if vt_total > 0 else 0.0
    risk_score = round(max(float(abuse_score), float(vt_ratio_score)), 1)

    # Classification Rules
    # Rule A: MALICIOUS
    if abuse_score >= 50 or vt_malicious >= 3 or abuse_reports >= 15:
        if abuse_score >= 50:
            reasons.append(f"High abuse confidence score ({abuse_score}%) reported on AbuseIPDB.")
        if vt_malicious >= 3:
            reasons.append(f"{vt_malicious} security vendors flagged this IP as malicious on VirusTotal.")
        if abuse_reports >= 15:
            reasons.append(f"Substantial community abuse reports ({abuse_reports} incidents).")

        return {
            "threat_status": "MALICIOUS",
            "risk_level": "CRITICAL" if (abuse_score >= 80 or vt_malicious >= 5) else "HIGH",
            "risk_score": max(50.0, risk_score),
            "is_malicious": True,
            "is_suspicious": False,
            "reasons": reasons,
            "assessment_summary": "Confirmed malicious threat activity detected across intelligence feeds."
        }

    # Rule B: SUSPICIOUS
    if (15 <= abuse_score < 50) or (1 <= vt_malicious < 3) or (vt_suspicious >= 2) or (abuse_reports > 0):
        if 15 <= abuse_score < 50:
            reasons.append(f"Moderate abuse confidence ({abuse_score}%) on AbuseIPDB.")
        if 1 <= vt_malicious < 3:
            reasons.append(f"{vt_malicious} security vendor(s) flagged this IP as malicious on VirusTotal.")
        if vt_suspicious >= 2:
            reasons.append(f"{vt_suspicious} security vendors marked this IP as suspicious on VirusTotal.")
        if abuse_reports > 0 and abuse_score < 15:
            reasons.append(f"Historical abuse reports detected ({abuse_reports} reports).")
        if is_tor:
            reasons.append("Identified as Tor Exit Node with observable network traffic reports.")

        return {
            "threat_status": "SUSPICIOUS",
            "risk_level": "MEDIUM",
            "risk_score": max(20.0, risk_score),
            "is_malicious": False,
            "is_suspicious": True,
            "reasons": reasons,
            "assessment_summary": "Suspicious indicators found. Recommended for active monitoring and traffic inspection."
        }

    # Rule C: Tor Exit Node alone with 0 abuse reports
    if is_tor and abuse_reports == 0 and vt_malicious == 0:
        return {
            "threat_status": "SUSPICIOUS",
            "risk_level": "LOW",
            "risk_score": 15.0,
            "is_malicious": False,
            "is_suspicious": True,
            "reasons": [
                "Identified as a public Tor Exit Node (privacy router). No malicious abuse reports currently registered."
            ],
            "assessment_summary": "Anonymization proxy detected. Tor routing alone does not establish malicious activity, but traffic warrants scrutiny."
        }

    # Rule D: NO KNOWN THREATS
    # Explicitly verify that feeds were successfully checked and returned zero malicious hits
    if (vt_available or abuse_available) and vt_malicious == 0 and abuse_score == 0 and abuse_reports == 0:
        feed_names = []
        if vt_available:
            feed_names.append("VirusTotal (0 detections)")
        if abuse_available:
            feed_names.append("AbuseIPDB (0% abuse confidence)")

        if is_whitelisted:
            reasons.append("IP address is verified on the global clean whitelist.")

        reasons.append(f"Verified clean across active feeds: {', '.join(feed_names)}.")

        return {
            "threat_status": "NO KNOWN THREATS",
            "risk_level": "SAFE",
            "risk_score": 0.0,
            "is_malicious": False,
            "is_suspicious": False,
            "reasons": reasons,
            "assessment_summary": "No known threat reports found across queried threat intelligence providers."
        }

    # Rule E: Fallback to UNKNOWN
    return {
        "threat_status": "UNKNOWN",
        "risk_level": "UNKNOWN",
        "risk_score": 0.0,
        "is_malicious": False,
        "is_suspicious": False,
        "reasons": ["Insufficient data available from threat intelligence providers to formulate a conclusive risk score."],
        "assessment_summary": "Intelligence inconclusive or not currently indexed."
    }


async def analyze_ip_threat(
    ip_address: str,
    force_refresh: bool = False,
    http_client: Optional[httpx.AsyncClient] = None
) -> Dict[str, Any]:
    """
    Main Threat Intelligence Orchestrator.
    
    1. Validates and categorizes the IP (IPv4/IPv6, Public/Private).
    2. Checks server-side cache.
    3. Concurrently queries VirusTotal v3 and AbuseIPDB.
    4. Evaluates risk using the hybrid classification engine.
    5. Caches and returns normalized result.
    """
    clean_ip = ip_address.strip() if ip_address else ""
    ip_meta = validate_and_classify_ip(clean_ip)

    if not ip_meta["is_valid"]:
        return {
            "success": False,
            "ip_address": clean_ip,
            "validation": ip_meta,
            "threat_status": "UNKNOWN",
            "risk_level": "INVALID",
            "risk_score": 0.0,
            "error": ip_meta["error"],
            "message": ip_meta["message"]
        }

    # Check cache if not force refreshing
    if not force_refresh:
        cached = threat_intel_cache.get(ip_meta["ip"])
        if cached:
            cached_copy = dict(cached)
            cached_copy["cached"] = True
            return cached_copy

    # If IP is not public routable, return instant local assessment without calling external APIs
    if not ip_meta["lookup_eligible"]:
        assessment = evaluate_hybrid_threat_status(ip_meta, {}, {})
        result = {
            "success": True,
            "ip_address": ip_meta["ip"],
            "ip_version": ip_meta["version"],
            "ip_type": ip_meta["ip_type"],
            "is_public": False,
            "is_private": ip_meta["is_private"],
            "cached": False,
            "lookup_timestamp": datetime.now(timezone.utc).isoformat(),
            "country": "Internal / Non-Routable",
            "isp": "Local / Reserved Network",
            "asn": 0,
            "asn_org": "N/A",
            "threat_status": assessment["threat_status"],
            "risk_level": assessment["risk_level"],
            "risk_score": assessment["risk_score"],
            "reasons": assessment["reasons"],
            "assessment_summary": assessment["assessment_summary"],
            "sources": {
                "virustotal": {"available": False, "status": "NOT_APPLICABLE", "message": "Non-public IP"},
                "abuseipdb": {"available": False, "status": "NOT_APPLICABLE", "message": "Non-public IP"}
            }
        }
        threat_intel_cache.set(ip_meta["ip"], result)
        return result

    # Query APIs concurrently with exception isolation
    if http_client is not None:
        vt_task = query_virustotal(ip_meta["ip"], http_client)
        abuse_task = query_abuseipdb(ip_meta["ip"], http_client)
        raw_results = await asyncio.gather(vt_task, abuse_task, return_exceptions=True)
    else:
        async with httpx.AsyncClient(headers={"User-Agent": "NIDS-Threat-Intelligence-System/2.0"}, trust_env=False) as client:
            vt_task = query_virustotal(ip_meta["ip"], client)
            abuse_task = query_abuseipdb(ip_meta["ip"], client)
            raw_results = await asyncio.gather(vt_task, abuse_task, return_exceptions=True)

    vt_res = raw_results[0] if isinstance(raw_results[0], dict) else {
        "source": "VirusTotal v3", "status": "ERROR", "available": False, "error": str(raw_results[0])
    }
    abuse_res = raw_results[1] if isinstance(raw_results[1], dict) else {
        "source": "AbuseIPDB", "status": "ERROR", "available": False, "error": str(raw_results[1])
    }

    # Evaluate threat assessment
    assessment = evaluate_hybrid_threat_status(ip_meta, vt_res, abuse_res)

    # Compile unified metadata
    country = abuse_res.get("country") or vt_res.get("country") or "Unknown"
    isp = abuse_res.get("isp") or vt_res.get("as_owner") or "Unknown"
    asn = vt_res.get("asn") or 0
    asn_org = vt_res.get("as_owner") or abuse_res.get("isp") or "Unknown"
    domain = abuse_res.get("domain") or "Unknown"

    result = {
        "success": True,
        "ip_address": ip_meta["ip"],
        "ip_version": ip_meta["version"],
        "ip_type": ip_meta["ip_type"],
        "is_public": True,
        "is_private": False,
        "cached": False,
        "lookup_timestamp": datetime.now(timezone.utc).isoformat(),
        "country": country,
        "country_name": abuse_res.get("country_name", country),
        "isp": isp,
        "domain": domain,
        "asn": asn,
        "asn_org": asn_org,
        "usage_type": abuse_res.get("usage_type", "Unknown"),
        "is_tor": abuse_res.get("is_tor", False),
        "threat_status": assessment["threat_status"],
        "risk_level": assessment["risk_level"],
        "risk_score": assessment["risk_score"],
        "reasons": assessment["reasons"],
        "assessment_summary": assessment["assessment_summary"],
        "virustotal": {
            "status": vt_res.get("status"),
            "available": vt_res.get("available", False),
            "malicious_count": vt_res.get("malicious_count", 0),
            "suspicious_count": vt_res.get("suspicious_count", 0),
            "total_engines": vt_res.get("total_engines", 0),
            "detection_ratio": vt_res.get("detection_ratio", "N/A"),
            "last_analysis_date": vt_res.get("last_analysis_date", "N/A"),
            "error": vt_res.get("error")
        },
        "abuseipdb": {
            "status": abuse_res.get("status"),
            "available": abuse_res.get("available", False),
            "abuse_confidence_score": abuse_res.get("abuse_confidence_score", 0),
            "total_reports": abuse_res.get("total_reports", 0),
            "last_reported_at": abuse_res.get("last_reported_at", "Never"),
            "is_whitelisted": abuse_res.get("is_whitelisted", False),
            "error": abuse_res.get("error")
        }
    }

    # Cache positive evaluations
    threat_intel_cache.set(ip_meta["ip"], result)
    return result
