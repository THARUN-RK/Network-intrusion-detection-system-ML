"""
Hybrid IP Threat Intelligence Service
Integrates VirusTotal API v3 and AbuseIPDB API v2 with:
- Secure runtime environment variable / .env loading (never hardcoded or exposed)
- Independent provider execution and fine-grained HTTP/network/schema error isolation
- Evidence-backed hybrid risk classification (BENIGN / NO KNOWN THREATS, SUSPICIOUS, MALICIOUS, UNKNOWN)
- Non-misleading risk scoring (None when no actionable intelligence is available)
"""

import os
import time
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

import httpx

import config
from backend.app.services.ip_validator import validate_and_classify_ip

logger = logging.getLogger(__name__)

# Endpoint URLs
VT_API_URL = "https://www.virustotal.com/api/v3/ip_addresses"
ABUSEIPDB_API_URL = "https://api.abuseipdb.com/api/v2/check"

# Cache TTL: 24 hours (86,400 seconds)
CACHE_TTL_SECONDS = int(os.getenv("THREAT_INTEL_CACHE_TTL", "86400"))

# Canonical Threat Statuses
THREAT_STATUS_NO_THREATS = "NO KNOWN THREATS"
THREAT_STATUS_SUSPICIOUS = "SUSPICIOUS"
THREAT_STATUS_MALICIOUS = "MALICIOUS"
THREAT_STATUS_UNKNOWN = "UNKNOWN"

# Track latest sanitized runtime provider connection states (no secrets or IPs stored)
_PROVIDER_RUNTIME_STATE: Dict[str, Dict[str, Any]] = {
    "virustotal": {"last_status": None, "connection_status": None, "last_error": None, "updated_at": None},
    "abuseipdb": {"last_status": None, "connection_status": None, "last_error": None, "updated_at": None},
}


def _safe_int(value: Any, default: int = 0) -> int:
    """Safely converts an arbitrary JSON value to int without raising."""
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely converts an arbitrary JSON value to float without raising."""
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any, default: str = "Unknown") -> str:
    """Safely converts an arbitrary JSON value to non-empty stripped str."""
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _safe_dict(value: Any) -> Dict[str, Any]:
    """Ensures JSON sub-objects default to an empty dict even when explicit null."""
    return value if isinstance(value, dict) else {}


def _safe_json_dict(response: httpx.Response, provider_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parses HTTP response JSON safely and ensures a top-level JSON object."""
    try:
        parsed = response.json()
    except (ValueError, TypeError) as exc:
        return None, f"{provider_name} returned invalid JSON response ({exc})."
    if not isinstance(parsed, dict):
        return None, f"{provider_name} returned unexpected JSON structure."
    return parsed, None


def _format_asn(raw_asn: Any) -> str:
    """Formats numeric or string ASN values uniformly."""
    if raw_asn is None or isinstance(raw_asn, bool):
        return "Unknown"
    if isinstance(raw_asn, int):
        return f"AS{raw_asn}" if raw_asn > 0 else "Unknown"
    text = str(raw_asn).strip()
    if not text or text == "0":
        return "Unknown"
    return text if text.upper().startswith("AS") else f"AS{text}"


def _format_timestamp_value(raw_val: Any, default: str = "N/A") -> str:
    """Formats UNIX epoch seconds or ISO-8601 timestamp strings to UTC display."""
    if raw_val is None:
        return default
    if isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool):
        try:
            return datetime.fromtimestamp(float(raw_val), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        except (OSError, OverflowError, ValueError):
            return default
    text = str(raw_val).strip()
    if not text:
        return default
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return text


def _record_provider_state(provider: str, status: str, connection_status: str, error: Optional[str]) -> None:
    """Records latest sanitized connection outcome for dashboard diagnostics."""
    if provider in _PROVIDER_RUNTIME_STATE:
        _PROVIDER_RUNTIME_STATE[provider] = {
            "last_status": status,
            "connection_status": connection_status,
            "last_error": error,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


def get_threat_intel_status() -> Dict[str, Any]:
    """
    Returns live configuration and connection status for VirusTotal and AbuseIPDB.
    Reads current environment dynamically without exposing secret API keys.
    """
    cfg = config.get_threat_intel_config_status()

    def _build_provider_state(key_name: str, cfg_entry: Dict[str, Any]) -> Dict[str, Any]:
        configured = bool(cfg_entry["configured"])
        runtime = _PROVIDER_RUNTIME_STATE.get(key_name, {})
        if not configured:
            return {
                "provider": cfg_entry["provider"],
                "env_var": cfg_entry["env_var"],
                "configured": False,
                "status": "NOT_CONFIGURED",
                "connection_status": "Not Configured",
                "last_error": f"{cfg_entry['env_var']} environment variable is not configured.",
                "updated_at": runtime.get("updated_at"),
            }

        last_status = runtime.get("last_status")
        if last_status and last_status != "NOT_CONFIGURED":
            return {
                "provider": cfg_entry["provider"],
                "env_var": cfg_entry["env_var"],
                "configured": True,
                "status": last_status,
                "connection_status": runtime.get("connection_status") or "Configured",
                "last_error": runtime.get("last_error"),
                "updated_at": runtime.get("updated_at"),
            }

        return {
            "provider": cfg_entry["provider"],
            "env_var": cfg_entry["env_var"],
            "configured": True,
            "status": "CONFIGURED",
            "connection_status": "Configured (Ready)",
            "last_error": None,
            "updated_at": None,
        }

    return {
        "dotenv_file_present": cfg["dotenv_file_present"],
        "virustotal": _build_provider_state("virustotal", cfg["virustotal"]),
        "abuseipdb": _build_provider_state("abuseipdb", cfg["abuseipdb"]),
    }


class ThreatIntelCache:
    """In-memory TTL cache for positive or local IP intelligence lookups."""
    def __init__(self, ttl_seconds: int = CACHE_TTL_SECONDS):
        self.ttl = ttl_seconds
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def get(self, ip: str) -> Optional[Dict[str, Any]]:
        if ip in self._cache:
            timestamp, data = self._cache[ip]
            if time.time() - timestamp < self.ttl:
                return data
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
    Queries official VirusTotal API v3 `/api/v3/ip_addresses/{ip}`.
    Retrieves vendor counts, country, ASN, network CIDR, and last analysis date.
    Never exposes the API key and handles every HTTP/network failure mode safely.
    """
    api_key = config.get_api_key("VIRUSTOTAL_API_KEY")
    if not api_key:
        err = "VIRUSTOTAL_API_KEY environment variable is not configured."
        _record_provider_state("virustotal", "NOT_CONFIGURED", "Not Configured", err)
        return {
            "source": "VirusTotal v3",
            "configured": False,
            "status": "NOT_CONFIGURED",
            "connection_status": "Not Configured",
            "available": False,
            "malicious_count": 0,
            "suspicious_count": 0,
            "harmless_count": 0,
            "undetected_count": 0,
            "total_engines": 0,
            "detection_ratio": "N/A",
            "country": "Unknown",
            "asn": "Unknown",
            "as_owner": "Unknown",
            "network": "Unknown",
            "last_analysis_date": "N/A",
            "error": err,
        }

    url = f"{VT_API_URL}/{ip}"
    headers = {
        "x-apikey": api_key,
        "Accept": "application/json",
    }

    try:
        response = await client.get(url, headers=headers, timeout=8.0)

        if response.status_code == 200:
            raw_json, json_err = _safe_json_dict(response, "VirusTotal v3")
            if json_err:
                _record_provider_state("virustotal", "INVALID_RESPONSE", "Invalid Response", json_err)
                return {
                    "source": "VirusTotal v3",
                    "configured": True,
                    "status": "INVALID_RESPONSE",
                    "connection_status": "Invalid Response",
                    "available": False,
                    "error": json_err,
                }

            data_obj = _safe_dict(raw_json.get("data"))
            attrs = _safe_dict(data_obj.get("attributes"))
            if not attrs and not data_obj:
                msg = "VirusTotal returned empty data payload (missing attributes)."
                _record_provider_state("virustotal", "INVALID_RESPONSE", "Malformed Payload", msg)
                return {
                    "source": "VirusTotal v3",
                    "configured": True,
                    "status": "INVALID_RESPONSE",
                    "connection_status": "Malformed Payload",
                    "available": False,
                    "error": msg,
                }

            stats = _safe_dict(attrs.get("last_analysis_stats"))
            malicious = max(0, _safe_int(stats.get("malicious"), 0))
            suspicious = max(0, _safe_int(stats.get("suspicious"), 0))
            harmless = max(0, _safe_int(stats.get("harmless"), 0))
            undetected = max(0, _safe_int(stats.get("undetected"), 0))
            timeout_cnt = max(0, _safe_int(stats.get("timeout"), 0))

            total_engines = malicious + suspicious + harmless + undetected + timeout_cnt
            if total_engines == 0:
                results_map = _safe_dict(attrs.get("last_analysis_results"))
                total_engines = len(results_map)

            last_date_str = _format_timestamp_value(attrs.get("last_analysis_date"), default="N/A")
            country = _safe_str(attrs.get("country"), "Unknown")
            asn = _format_asn(attrs.get("asn"))
            as_owner = _safe_str(attrs.get("as_owner"), "Unknown")
            network = _safe_str(attrs.get("network"), "Unknown")

            _record_provider_state("virustotal", "SUCCESS", "Connected", None)
            return {
                "source": "VirusTotal v3",
                "configured": True,
                "status": "SUCCESS",
                "connection_status": "Connected",
                "available": True,
                "malicious_count": malicious,
                "suspicious_count": suspicious,
                "harmless_count": harmless,
                "undetected_count": undetected,
                "total_engines": total_engines,
                "detection_ratio": f"{malicious}/{total_engines}" if total_engines > 0 else "0/0",
                "country": country,
                "asn": asn,
                "as_owner": as_owner,
                "network": network,
                "last_analysis_date": last_date_str,
                "reputation_raw": _safe_int(attrs.get("reputation"), 0),
                "error": None,
            }

        if response.status_code == 404:
            _record_provider_state("virustotal", "NO_RECORD", "Connected (Not Indexed)", None)
            return {
                "source": "VirusTotal v3",
                "configured": True,
                "status": "NO_RECORD",
                "connection_status": "Connected (Not Indexed)",
                "available": True,
                "malicious_count": 0,
                "suspicious_count": 0,
                "harmless_count": 0,
                "undetected_count": 0,
                "total_engines": 0,
                "detection_ratio": "0/0",
                "country": "Unknown",
                "asn": "Unknown",
                "as_owner": "Unknown",
                "network": "Unknown",
                "last_analysis_date": "N/A",
                "message": "IP address is not currently indexed in VirusTotal database.",
                "error": None,
            }

        if response.status_code == 401:
            err = "Invalid or revoked VirusTotal API key (HTTP 401 Unauthorized)."
            _record_provider_state("virustotal", "UNAUTHORIZED", "Invalid API Key (HTTP 401)", err)
            return {
                "source": "VirusTotal v3",
                "configured": True,
                "status": "UNAUTHORIZED",
                "connection_status": "Invalid API Key (HTTP 401)",
                "available": False,
                "error": err,
            }

        if response.status_code == 403:
            err = "VirusTotal API access forbidden or insufficient tier permissions (HTTP 403 Forbidden)."
            _record_provider_state("virustotal", "FORBIDDEN", "Access Forbidden (HTTP 403)", err)
            return {
                "source": "VirusTotal v3",
                "configured": True,
                "status": "FORBIDDEN",
                "connection_status": "Access Forbidden (HTTP 403)",
                "available": False,
                "error": err,
            }

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            hint = f" Retry after {retry_after}s." if retry_after else " (Public tier limit: 4 requests/min)."
            err = f"VirusTotal API rate limit exceeded (HTTP 429).{hint}"
            logger.warning(f"[VirusTotal] Rate limit exceeded querying {ip}")
            _record_provider_state("virustotal", "RATE_LIMITED", "Rate Limited (HTTP 429)", err)
            return {
                "source": "VirusTotal v3",
                "configured": True,
                "status": "RATE_LIMITED",
                "connection_status": "Rate Limited (HTTP 429)",
                "available": False,
                "error": err,
            }

        if 500 <= response.status_code < 600:
            err = f"VirusTotal upstream server failure (HTTP {response.status_code})."
            _record_provider_state("virustotal", "SERVER_ERROR", f"Server Error (HTTP {response.status_code})", err)
            return {
                "source": "VirusTotal v3",
                "configured": True,
                "status": "SERVER_ERROR",
                "connection_status": f"Server Error (HTTP {response.status_code})",
                "available": False,
                "error": err,
            }

        err = f"VirusTotal API returned HTTP {response.status_code}."
        _record_provider_state("virustotal", "ERROR", f"HTTP Error ({response.status_code})", err)
        return {
            "source": "VirusTotal v3",
            "configured": True,
            "status": "ERROR",
            "connection_status": f"HTTP Error ({response.status_code})",
            "available": False,
            "error": err,
        }

    except httpx.TimeoutException:
        err = "VirusTotal API request timed out after 8.0s."
        logger.warning(f"[VirusTotal] Request timed out for {ip}")
        _record_provider_state("virustotal", "TIMEOUT", "Timed Out", err)
        return {
            "source": "VirusTotal v3",
            "configured": True,
            "status": "TIMEOUT",
            "connection_status": "Timed Out",
            "available": False,
            "error": err,
        }
    except httpx.RequestError as exc:
        err = f"Network error connecting to VirusTotal: {type(exc).__name__}"
        logger.error(f"[VirusTotal] Connection failure for {ip}: {exc}")
        _record_provider_state("virustotal", "CONNECTION_ERROR", "Connection Error", err)
        return {
            "source": "VirusTotal v3",
            "configured": True,
            "status": "CONNECTION_ERROR",
            "connection_status": "Connection Error",
            "available": False,
            "error": err,
        }
    except Exception as exc:
        err = f"Unexpected VirusTotal error: {type(exc).__name__}"
        logger.error(f"[VirusTotal] Unexpected error for {ip}: {exc}", exc_info=True)
        _record_provider_state("virustotal", "ERROR", "Error", err)
        return {
            "source": "VirusTotal v3",
            "configured": True,
            "status": "ERROR",
            "connection_status": "Error",
            "available": False,
            "error": err,
        }


async def query_abuseipdb(ip: str, client: httpx.AsyncClient) -> Dict[str, Any]:
    """
    Queries official AbuseIPDB API v2 `/api/v2/check`.
    Retrieves abuse confidence score, total reports, distinct reporters,
    country, ISP, domain, usage type, Tor flag, and last reported timestamp.
    """
    api_key = config.get_api_key("ABUSEIPDB_API_KEY")
    if not api_key:
        err = "ABUSEIPDB_API_KEY environment variable is not configured."
        _record_provider_state("abuseipdb", "NOT_CONFIGURED", "Not Configured", err)
        return {
            "source": "AbuseIPDB",
            "configured": False,
            "status": "NOT_CONFIGURED",
            "connection_status": "Not Configured",
            "available": False,
            "abuse_confidence_score": 0,
            "total_reports": 0,
            "num_distinct_users": 0,
            "country": "Unknown",
            "country_name": "Unknown",
            "isp": "Unknown",
            "domain": "Unknown",
            "usage_type": "Unknown",
            "is_tor": False,
            "is_whitelisted": False,
            "last_reported_at": "Never",
            "error": err,
        }

    headers = {
        "Key": api_key,
        "Accept": "application/json",
    }
    params = {
        "ipAddress": ip,
        "maxAgeInDays": "90",
        "verbose": "true",
    }

    try:
        response = await client.get(ABUSEIPDB_API_URL, headers=headers, params=params, timeout=8.0)

        if response.status_code == 200:
            raw_json, json_err = _safe_json_dict(response, "AbuseIPDB")
            if json_err:
                _record_provider_state("abuseipdb", "INVALID_RESPONSE", "Invalid Response", json_err)
                return {
                    "source": "AbuseIPDB",
                    "configured": True,
                    "status": "INVALID_RESPONSE",
                    "connection_status": "Invalid Response",
                    "available": False,
                    "error": json_err,
                }

            payload = _safe_dict(raw_json.get("data"))
            if not payload:
                msg = "AbuseIPDB returned empty data payload (missing 'data' fields)."
                _record_provider_state("abuseipdb", "INVALID_RESPONSE", "Malformed Payload", msg)
                return {
                    "source": "AbuseIPDB",
                    "configured": True,
                    "status": "INVALID_RESPONSE",
                    "connection_status": "Malformed Payload",
                    "available": False,
                    "error": msg,
                }

            abuse_score = max(0, min(100, _safe_int(payload.get("abuseConfidenceScore"), 0)))
            total_reports = max(0, _safe_int(payload.get("totalReports"), 0))
            distinct_users = max(0, _safe_int(payload.get("numDistinctUsers"), 0))
            country = _safe_str(payload.get("countryCode"), "Unknown")
            country_name = _safe_str(payload.get("countryName"), country)
            isp = _safe_str(payload.get("isp"), "Unknown")
            domain = _safe_str(payload.get("domain"), "Unknown")
            usage_type = _safe_str(payload.get("usageType"), "Unknown")
            is_tor = bool(payload.get("isTor", False))
            is_whitelisted = bool(payload.get("isWhitelisted", False))
            last_reported = _format_timestamp_value(payload.get("lastReportedAt"), default="Never")

            _record_provider_state("abuseipdb", "SUCCESS", "Connected", None)
            return {
                "source": "AbuseIPDB",
                "configured": True,
                "status": "SUCCESS",
                "connection_status": "Connected",
                "available": True,
                "abuse_confidence_score": abuse_score,
                "total_reports": total_reports,
                "num_distinct_users": distinct_users,
                "country": country,
                "country_name": country_name,
                "isp": isp,
                "domain": domain,
                "usage_type": usage_type,
                "is_tor": is_tor,
                "is_whitelisted": is_whitelisted,
                "last_reported_at": last_reported,
                "error": None,
            }

        if response.status_code == 401:
            err = "Invalid or unauthorized AbuseIPDB API key (HTTP 401 Unauthorized)."
            _record_provider_state("abuseipdb", "UNAUTHORIZED", "Invalid API Key (HTTP 401)", err)
            return {
                "source": "AbuseIPDB",
                "configured": True,
                "status": "UNAUTHORIZED",
                "connection_status": "Invalid API Key (HTTP 401)",
                "available": False,
                "error": err,
            }

        if response.status_code == 403:
            err = "AbuseIPDB API access forbidden or insufficient permissions (HTTP 403 Forbidden)."
            _record_provider_state("abuseipdb", "FORBIDDEN", "Access Forbidden (HTTP 403)", err)
            return {
                "source": "AbuseIPDB",
                "configured": True,
                "status": "FORBIDDEN",
                "connection_status": "Access Forbidden (HTTP 403)",
                "available": False,
                "error": err,
            }

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            hint = f" Retry after {retry_after}s." if retry_after else " (Free tier limit: 1,000 checks/day)."
            err = f"AbuseIPDB daily API rate limit reached (HTTP 429).{hint}"
            logger.warning(f"[AbuseIPDB] Rate limit exceeded querying {ip}")
            _record_provider_state("abuseipdb", "RATE_LIMITED", "Rate Limited (HTTP 429)", err)
            return {
                "source": "AbuseIPDB",
                "configured": True,
                "status": "RATE_LIMITED",
                "connection_status": "Rate Limited (HTTP 429)",
                "available": False,
                "error": err,
            }

        if 500 <= response.status_code < 600:
            err = f"AbuseIPDB upstream server failure (HTTP {response.status_code})."
            _record_provider_state("abuseipdb", "SERVER_ERROR", f"Server Error (HTTP {response.status_code})", err)
            return {
                "source": "AbuseIPDB",
                "configured": True,
                "status": "SERVER_ERROR",
                "connection_status": f"Server Error (HTTP {response.status_code})",
                "available": False,
                "error": err,
            }

        err = f"AbuseIPDB API returned HTTP {response.status_code}."
        _record_provider_state("abuseipdb", "ERROR", f"HTTP Error ({response.status_code})", err)
        return {
            "source": "AbuseIPDB",
            "configured": True,
            "status": "ERROR",
            "connection_status": f"HTTP Error ({response.status_code})",
            "available": False,
            "error": err,
        }

    except httpx.TimeoutException:
        err = "AbuseIPDB API request timed out after 8.0s."
        logger.warning(f"[AbuseIPDB] Request timed out for {ip}")
        _record_provider_state("abuseipdb", "TIMEOUT", "Timed Out", err)
        return {
            "source": "AbuseIPDB",
            "configured": True,
            "status": "TIMEOUT",
            "connection_status": "Timed Out",
            "available": False,
            "error": err,
        }
    except httpx.RequestError as exc:
        err = f"Network error connecting to AbuseIPDB: {type(exc).__name__}"
        logger.error(f"[AbuseIPDB] Connection failure for {ip}: {exc}")
        _record_provider_state("abuseipdb", "CONNECTION_ERROR", "Connection Error", err)
        return {
            "source": "AbuseIPDB",
            "configured": True,
            "status": "CONNECTION_ERROR",
            "connection_status": "Connection Error",
            "available": False,
            "error": err,
        }
    except Exception as exc:
        err = f"Unexpected AbuseIPDB error: {type(exc).__name__}"
        logger.error(f"[AbuseIPDB] Unexpected error for {ip}: {exc}", exc_info=True)
        _record_provider_state("abuseipdb", "ERROR", "Error", err)
        return {
            "source": "AbuseIPDB",
            "configured": True,
            "status": "ERROR",
            "connection_status": "Error",
            "available": False,
            "error": err,
        }


def _describe_unavailable_feed(name: str, res: Dict[str, Any]) -> str:
    """Formats a clear diagnostic string when one feed did not contribute."""
    err = res.get("error") or res.get("message") or res.get("connection_status") or res.get("status") or "Unavailable"
    return f"{name}: {err}"


def evaluate_hybrid_threat_status(
    ip_meta: Dict[str, Any],
    vt_res: Dict[str, Any],
    abuse_res: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evidence-Based Hybrid Threat Classification Engine.

    Produces one of 4 canonical classifications:
    - BENIGN / NO KNOWN THREATS (status key: "NO KNOWN THREATS")
    - SUSPICIOUS
    - MALICIOUS
    - UNKNOWN

    Core Safety Rules:
    - Never classifies an IP as benign/safe when both threat feeds are unavailable or unconfigured.
    - Returns risk_score = None (risk_score_available = False) when no actionable intelligence exists,
      preventing misleading "0/100" safe scores.
    - Evaluates single active API results independently if the other API fails.
    - Never scores or classifies an IP from its numerical address alone.
    """
    vt_res = _safe_dict(vt_res)
    abuse_res = _safe_dict(abuse_res)

    # 0. Invalid IP address
    if not ip_meta.get("is_valid", True):
        err_msg = ip_meta.get("error") or "Provided value is not a valid IPv4 or IPv6 address."
        return {
            "threat_status": THREAT_STATUS_UNKNOWN,
            "classification_label": THREAT_STATUS_UNKNOWN,
            "risk_level": "INVALID",
            "risk_score": None,
            "risk_score_available": False,
            "is_malicious": False,
            "is_suspicious": False,
            "reasons": [err_msg],
            "evidence_summary": f"Invalid IP address input: {err_msg}",
            "assessment_summary": "Enter a valid public IPv4 or IPv6 address to query external threat intelligence.",
            "recommended_action": "Enter a valid single IPv4 or IPv6 host address (without port numbers or CIDR suffixes).",
        }

    # 1. Non-public IP addresses (Private, Loopback, Link-Local, Reserved, CGNAT, Documentation)
    if not ip_meta.get("is_public", False):
        ip_type = ip_meta.get("ip_type", "NON_PUBLIC")
        return {
            "threat_status": THREAT_STATUS_UNKNOWN,
            "classification_label": THREAT_STATUS_UNKNOWN,
            "risk_level": "INFO",
            "risk_score": None,
            "risk_score_available": False,
            "is_malicious": False,
            "is_suspicious": False,
            "reasons": [
                f"{ip_type} range cannot be verified against public threat feeds (RFC-reserved / internal scope)."
            ],
            "evidence_summary": (
                f"{ip_meta.get('ip', '')} is classified as {ip_type}. External threat intelligence providers "
                "only track globally routable public IPs."
            ),
            "assessment_summary": f"Internal or non-routable IP address ({ip_type}). External reputation lookup was skipped safely.",
            "recommended_action": (
                "No external threat lookup applies to internal network ranges. Use the IP-Based Attack Analyzer "
                "or SVM Flow Classifier to inspect local packet behavior."
            ),
        }

    # 2. Check actionable intelligence availability
    vt_status = vt_res.get("status")
    vt_total_engines = _safe_int(vt_res.get("total_engines"), 0)
    # Require actual vendor inspection results (SUCCESS) or total_engines > 0 from VT;
    # a 404 NO_RECORD with 0 engines alone cannot prove an IP is safe without AbuseIPDB.
    vt_usable = bool(vt_res.get("available", False)) and (
        vt_status == "SUCCESS" or (vt_status == "NO_RECORD" and vt_total_engines > 0)
    )
    abuse_usable = bool(abuse_res.get("available", False)) and abuse_res.get("status") == "SUCCESS"

    if not vt_usable and not abuse_usable:
        err_details: List[str] = []
        if vt_status == "NO_RECORD":
            err_details.append("VirusTotal v3: IP not indexed in VirusTotal (0 vendor scans available).")
        elif vt_res:
            err_details.append(_describe_unavailable_feed("VirusTotal v3", vt_res))
        else:
            err_details.append("VirusTotal v3: No telemetry received.")

        if abuse_res:
            err_details.append(_describe_unavailable_feed("AbuseIPDB v2", abuse_res))
        else:
            err_details.append("AbuseIPDB v2: No telemetry received.")

        reasons = ["No actionable threat intelligence is currently available from VirusTotal or AbuseIPDB."] + err_details

        both_unconfigured = (
            vt_res.get("status") == "NOT_CONFIGURED" and abuse_res.get("status") == "NOT_CONFIGURED"
        )
        if both_unconfigured:
            action = (
                "Configure VIRUSTOTAL_API_KEY and ABUSEIPDB_API_KEY in Render Environment Variables "
                "or local .env to enable live threat intelligence."
            )
            summary = "Insufficient intelligence: Threat intelligence API keys are not configured."
        else:
            action = (
                "Check API key validity, tier rate limits, and outbound network connectivity for VirusTotal / AbuseIPDB, "
                "then retry with Bypass Cache enabled."
            )
            summary = "Insufficient intelligence: Threat intelligence providers failed to return actionable telemetry."

        return {
            "threat_status": THREAT_STATUS_UNKNOWN,
            "classification_label": THREAT_STATUS_UNKNOWN,
            "risk_level": "UNKNOWN",
            "risk_score": None,
            "risk_score_available": False,
            "is_malicious": False,
            "is_suspicious": False,
            "reasons": reasons,
            "evidence_summary": "; ".join(err_details),
            "assessment_summary": summary,
            "recommended_action": action,
        }

    # 3. Extract metrics safely from whichever feed(s) succeeded
    vt_malicious = max(0, _safe_int(vt_res.get("malicious_count"), 0)) if vt_usable else 0
    vt_suspicious = max(0, _safe_int(vt_res.get("suspicious_count"), 0)) if vt_usable else 0
    vt_harmless = max(0, _safe_int(vt_res.get("harmless_count"), 0)) if vt_usable else 0
    vt_undetected = max(0, _safe_int(vt_res.get("undetected_count"), 0)) if vt_usable else 0
    vt_total = max(0, vt_total_engines) if vt_usable else 0

    abuse_score = max(0.0, min(100.0, _safe_float(abuse_res.get("abuse_confidence_score"), 0.0))) if abuse_usable else 0.0
    abuse_reports = max(0, _safe_int(abuse_res.get("total_reports"), 0)) if abuse_usable else 0
    abuse_distinct = max(0, _safe_int(abuse_res.get("num_distinct_users"), 0)) if abuse_usable else 0
    is_tor = bool(abuse_res.get("is_tor", False)) if abuse_usable else False
    is_whitelisted = bool(abuse_res.get("is_whitelisted", False)) if abuse_usable else False

    # Construct evidence notes for partial coverage when one provider failed/unconfigured
    partial_notes: List[str] = []
    if vt_usable and not abuse_usable:
        partial_notes.append(f"Partial feed coverage — {_describe_unavailable_feed('AbuseIPDB v2', abuse_res)}")
    elif abuse_usable and not vt_usable:
        if vt_status == "NO_RECORD":
            partial_notes.append("VirusTotal v3 had no indexed record for this IP; classified using AbuseIPDB telemetry.")
        else:
            partial_notes.append(f"Partial feed coverage — {_describe_unavailable_feed('VirusTotal v3', vt_res)}")

    # Compute base risk score from available telemetry (0 - 100)
    vt_ratio_score = (vt_malicious / max(1, vt_total)) * 100.0 if (vt_usable and vt_total > 0) else 0.0
    raw_risk_score = round(max(float(abuse_score), float(vt_ratio_score)), 1)

    evidence: List[str] = []

    # Rule A: MALICIOUS
    if abuse_score >= 50 or vt_malicious >= 3 or abuse_reports >= 15:
        if abuse_usable and abuse_score >= 50:
            evidence.append(f"High abuse confidence score ({int(abuse_score)}%) reported on AbuseIPDB.")
        if vt_usable and vt_malicious >= 3:
            evidence.append(f"{vt_malicious}/{vt_total} security vendors flagged this IP as malicious on VirusTotal.")
        if abuse_usable and abuse_reports >= 15:
            evidence.append(
                f"Substantial community abuse reports ({abuse_reports} reports across {abuse_distinct} distinct reporters)."
            )
        if is_tor:
            evidence.append("Flagged as an active Tor Exit Node in AbuseIPDB.")

        reasons = evidence + partial_notes
        final_score = round(min(100.0, max(50.0, raw_risk_score)), 1)
        level = "CRITICAL" if (abuse_score >= 80 or vt_malicious >= 5) else "HIGH"
        action = (
            "Block this IP at perimeter firewall/WAF immediately, terminate any active connections, "
            "and audit historical flows in the IP-Based Attack Analyzer."
        )
        return {
            "threat_status": THREAT_STATUS_MALICIOUS,
            "classification_label": THREAT_STATUS_MALICIOUS,
            "risk_level": level,
            "risk_score": final_score,
            "risk_score_available": True,
            "is_malicious": True,
            "is_suspicious": False,
            "reasons": reasons,
            "evidence_summary": " ".join(evidence),
            "assessment_summary": "Confirmed malicious threat indicators detected across active intelligence feeds.",
            "recommended_action": action,
        }

    # Rule B: SUSPICIOUS
    # Require >= 3 abuse reports when abuse confidence is < 15% so 1-2 stale reports
    # on shared Mobile ISP / CGNAT IPs with 0% confidence do not trigger false alarms.
    if (15 <= abuse_score < 50) or (1 <= vt_malicious < 3) or (vt_suspicious >= 2) or (abuse_reports >= 3):
        if abuse_usable and 15 <= abuse_score < 50:
            evidence.append(f"Moderate abuse confidence ({int(abuse_score)}%) on AbuseIPDB.")
        if vt_usable and 1 <= vt_malicious < 3:
            evidence.append(f"{vt_malicious}/{vt_total} security vendor(s) flagged this IP as malicious on VirusTotal.")
        if vt_usable and vt_suspicious >= 2:
            evidence.append(f"{vt_suspicious}/{vt_total} security vendors marked this IP as suspicious on VirusTotal.")
        if abuse_usable and abuse_reports >= 3 and abuse_score < 15:
            evidence.append(f"Repeated abuse activity recorded on AbuseIPDB ({abuse_reports} report(s)).")
        if is_tor:
            evidence.append("Identified as a Tor Exit Node with observable traffic reports.")

        reasons = evidence + partial_notes
        final_score = round(min(49.9, max(20.0, raw_risk_score)), 1)
        action = (
            "Increase monitoring and apply rate-limiting policies; inspect packets from this source using "
            "the SVM Flow Classifier."
        )
        return {
            "threat_status": THREAT_STATUS_SUSPICIOUS,
            "classification_label": THREAT_STATUS_SUSPICIOUS,
            "risk_level": "MEDIUM",
            "risk_score": final_score,
            "risk_score_available": True,
            "is_malicious": False,
            "is_suspicious": True,
            "reasons": reasons,
            "evidence_summary": " ".join(evidence),
            "assessment_summary": "Suspicious indicators found. Recommended for active monitoring and traffic inspection.",
            "recommended_action": action,
        }

    # Rule C: Tor Exit Node alone with 0 abuse reports and 0 malicious detections
    if is_tor and abuse_reports == 0 and vt_malicious == 0:
        evidence.append("Identified as a public Tor Exit Node (anonymization proxy). No abuse reports registered.")
        reasons = evidence + partial_notes
        return {
            "threat_status": THREAT_STATUS_SUSPICIOUS,
            "classification_label": THREAT_STATUS_SUSPICIOUS,
            "risk_level": "LOW",
            "risk_score": 15.0,
            "risk_score_available": True,
            "is_malicious": False,
            "is_suspicious": True,
            "reasons": reasons,
            "evidence_summary": " ".join(evidence),
            "assessment_summary": "Anonymization proxy detected. Tor routing alone does not establish malicious activity, but warrants scrutiny.",
            "recommended_action": "Monitor connections originating from anonymization relays against sensitive authentication endpoints.",
        }

    # Rule D: BENIGN / NO KNOWN THREATS (at least one active feed verified clean and 0 malicious hits)
    feed_summaries: List[str] = []
    if vt_usable:
        feed_summaries.append(f"VirusTotal v3 ({vt_malicious}/{vt_total} malicious, {vt_harmless} harmless, {vt_undetected} undetected)")
    if abuse_usable:
        feed_summaries.append(f"AbuseIPDB v2 ({int(abuse_score)}% abuse confidence, {abuse_reports} reports)")

    if is_whitelisted:
        evidence.append("IP address appears on AbuseIPDB's global clean whitelist.")
    evidence.append(f"Verified clean across active intelligence feeds: {', '.join(feed_summaries)}.")
    if abuse_usable and 0 < abuse_reports < 3 and abuse_score < 15:
        evidence.append(
            f"Note: {abuse_reports} low-confidence historical report(s) ({int(abuse_score)}% abuse confidence), "
            "typical of shared ISP / mobile CGNAT rotation."
        )

    reasons = evidence + partial_notes
    return {
        "threat_status": THREAT_STATUS_NO_THREATS,
        "classification_label": "BENIGN / NO KNOWN THREATS",
        "risk_level": "SAFE",
        "risk_score": 0.0,
        "risk_score_available": True,
        "is_malicious": False,
        "is_suspicious": False,
        "reasons": reasons,
        "evidence_summary": " ".join(evidence),
        "assessment_summary": "No known threats detected across queried threat intelligence feeds.",
        "recommended_action": "No containment action required. Continue standard passive network monitoring.",
    }


async def analyze_ip_threat(
    ip_address: str,
    force_refresh: bool = False,
    http_client: Optional[httpx.AsyncClient] = None
) -> Dict[str, Any]:
    """
    Main Threat Intelligence Orchestrator.

    1. Validates and categorizes IPv4/IPv6 scope.
    2. Checks TTL cache for complete prior results.
    3. Concurrently queries VirusTotal v3 and AbuseIPDB v2 with fault isolation.
    4. Classifies risk cleanly without ever inventing intelligence or defaulting errors to 0/100 safe.
    5. Caches positive/local results without poisoning cache with transient failures.
    """
    clean_ip = ip_address.strip() if isinstance(ip_address, str) else ""
    ip_meta = validate_and_classify_ip(clean_ip)
    checked_at = datetime.now(timezone.utc).isoformat()

    # Invalid IP path (full structured response)
    if not ip_meta["is_valid"]:
        assessment = evaluate_hybrid_threat_status(ip_meta, {}, {})
        cfg_status = get_threat_intel_status()
        na_vt = {
            "source": "VirusTotal v3",
            "configured": cfg_status["virustotal"]["configured"],
            "status": "NOT_APPLICABLE",
            "connection_status": "Skipped (Invalid IP)",
            "available": False,
            "malicious_count": 0,
            "suspicious_count": 0,
            "harmless_count": 0,
            "undetected_count": 0,
            "total_engines": 0,
            "detection_ratio": "N/A",
            "country": "Unknown",
            "asn": "Unknown",
            "as_owner": "Unknown",
            "network": "Unknown",
            "last_analysis_date": "N/A",
            "error": ip_meta.get("error"),
        }
        na_abuse = {
            "source": "AbuseIPDB",
            "configured": cfg_status["abuseipdb"]["configured"],
            "status": "NOT_APPLICABLE",
            "connection_status": "Skipped (Invalid IP)",
            "available": False,
            "abuse_confidence_score": 0,
            "total_reports": 0,
            "num_distinct_users": 0,
            "country": "Unknown",
            "country_name": "Unknown",
            "isp": "Unknown",
            "domain": "Unknown",
            "usage_type": "Unknown",
            "is_tor": False,
            "is_whitelisted": False,
            "last_reported_at": "Never",
            "error": ip_meta.get("error"),
        }
        return {
            "success": False,
            "ip_address": clean_ip,
            "ip_version": 0,
            "ip_type": "INVALID",
            "is_public": False,
            "is_private": False,
            "cached": False,
            "lookup_timestamp": checked_at,
            "validation": ip_meta,
            "country": "Unknown",
            "country_name": "Unknown",
            "isp": "Unknown",
            "domain": "Unknown",
            "asn": "Unknown",
            "as_owner": "Unknown",
            "usage_type": "Unknown",
            "is_tor": False,
            "threat_status": assessment["threat_status"],
            "classification_label": assessment["classification_label"],
            "risk_level": assessment["risk_level"],
            "risk_score": assessment["risk_score"],
            "risk_score_available": assessment["risk_score_available"],
            "reasons": assessment["reasons"],
            "evidence_summary": assessment["evidence_summary"],
            "assessment_summary": assessment["assessment_summary"],
            "recommended_action": assessment["recommended_action"],
            "virustotal": na_vt,
            "abuseipdb": na_abuse,
            "api_status": cfg_status,
            "error": ip_meta["error"],
            "message": ip_meta["message"],
        }

    # Check cache if not force refreshing
    if not force_refresh:
        cached = threat_intel_cache.get(ip_meta["ip"])
        if cached:
            cached_copy = dict(cached)
            cached_copy["cached"] = True
            return cached_copy

    # Non-public IP path (Private, Loopback, Link-local, Reserved, etc.)
    if not ip_meta["lookup_eligible"]:
        assessment = evaluate_hybrid_threat_status(ip_meta, {}, {})
        cfg_status = get_threat_intel_status()
        local_vt = {
            "source": "VirusTotal v3",
            "configured": cfg_status["virustotal"]["configured"],
            "status": "NOT_APPLICABLE",
            "connection_status": f"Skipped ({ip_meta['ip_type']})",
            "available": False,
            "malicious_count": 0,
            "suspicious_count": 0,
            "harmless_count": 0,
            "undetected_count": 0,
            "total_engines": 0,
            "detection_ratio": "N/A",
            "country": "Internal / Non-Routable",
            "asn": "N/A",
            "as_owner": "Local / Reserved Network",
            "network": "N/A",
            "last_analysis_date": "N/A",
            "message": f"Skipped external lookup for {ip_meta['ip_type']} address.",
            "error": None,
        }
        local_abuse = {
            "source": "AbuseIPDB",
            "configured": cfg_status["abuseipdb"]["configured"],
            "status": "NOT_APPLICABLE",
            "connection_status": f"Skipped ({ip_meta['ip_type']})",
            "available": False,
            "abuse_confidence_score": 0,
            "total_reports": 0,
            "num_distinct_users": 0,
            "country": "Internal / Non-Routable",
            "country_name": "Internal / Non-Routable",
            "isp": "Local / Reserved Network",
            "domain": "N/A",
            "usage_type": ip_meta["ip_type"],
            "is_tor": False,
            "is_whitelisted": False,
            "last_reported_at": "Never",
            "message": f"Skipped external lookup for {ip_meta['ip_type']} address.",
            "error": None,
        }
        result = {
            "success": True,
            "ip_address": ip_meta["ip"],
            "ip_version": ip_meta["version"],
            "ip_type": ip_meta["ip_type"],
            "is_public": False,
            "is_private": ip_meta["is_private"],
            "cached": False,
            "lookup_timestamp": checked_at,
            "validation": ip_meta,
            "country": "Internal / Non-Routable",
            "country_name": "Internal / Non-Routable",
            "isp": "Local / Reserved Network",
            "domain": "N/A",
            "asn": "N/A",
            "as_owner": "Local / Reserved Network",
            "usage_type": ip_meta["ip_type"],
            "is_tor": False,
            "threat_status": assessment["threat_status"],
            "classification_label": assessment["classification_label"],
            "risk_level": assessment["risk_level"],
            "risk_score": assessment["risk_score"],
            "risk_score_available": assessment["risk_score_available"],
            "reasons": assessment["reasons"],
            "evidence_summary": assessment["evidence_summary"],
            "assessment_summary": assessment["assessment_summary"],
            "recommended_action": assessment["recommended_action"],
            "virustotal": local_vt,
            "abuseipdb": local_abuse,
            "api_status": cfg_status,
        }
        threat_intel_cache.set(ip_meta["ip"], result)
        return result

    # Query APIs concurrently with exception isolation
    if http_client is not None:
        vt_task = query_virustotal(ip_meta["ip"], http_client)
        abuse_task = query_abuseipdb(ip_meta["ip"], http_client)
        raw_results = await asyncio.gather(vt_task, abuse_task, return_exceptions=True)
    else:
        async with httpx.AsyncClient(
            headers={"User-Agent": "NIDS-Threat-Intelligence-System/2.0"},
            trust_env=False
        ) as client:
            vt_task = query_virustotal(ip_meta["ip"], client)
            abuse_task = query_abuseipdb(ip_meta["ip"], client)
            raw_results = await asyncio.gather(vt_task, abuse_task, return_exceptions=True)

    vt_res = raw_results[0] if isinstance(raw_results[0], dict) else {
        "source": "VirusTotal v3",
        "configured": bool(config.get_api_key("VIRUSTOTAL_API_KEY")),
        "status": "ERROR",
        "connection_status": "Exception Isolated",
        "available": False,
        "error": str(raw_results[0]),
    }
    abuse_res = raw_results[1] if isinstance(raw_results[1], dict) else {
        "source": "AbuseIPDB",
        "configured": bool(config.get_api_key("ABUSEIPDB_API_KEY")),
        "status": "ERROR",
        "connection_status": "Exception Isolated",
        "available": False,
        "error": str(raw_results[1]),
    }

    # Ensure mocked/external dicts always have baseline keys normalized
    vt_norm = {
        "source": "VirusTotal v3",
        "configured": bool(vt_res.get("configured", bool(config.get_api_key("VIRUSTOTAL_API_KEY")) or vt_res.get("available", False))),
        "status": vt_res.get("status", "UNKNOWN"),
        "connection_status": vt_res.get("connection_status") or ("Connected" if vt_res.get("available") else vt_res.get("status", "Unavailable")),
        "available": bool(vt_res.get("available", False)),
        "malicious_count": max(0, _safe_int(vt_res.get("malicious_count"), 0)),
        "suspicious_count": max(0, _safe_int(vt_res.get("suspicious_count"), 0)),
        "harmless_count": max(0, _safe_int(vt_res.get("harmless_count"), 0)),
        "undetected_count": max(0, _safe_int(vt_res.get("undetected_count"), 0)),
        "total_engines": max(0, _safe_int(vt_res.get("total_engines"), 0)),
        "detection_ratio": _safe_str(vt_res.get("detection_ratio"), "N/A"),
        "country": _safe_str(vt_res.get("country"), "Unknown"),
        "asn": _format_asn(vt_res.get("asn")),
        "as_owner": _safe_str(vt_res.get("as_owner"), "Unknown"),
        "network": _safe_str(vt_res.get("network"), "Unknown"),
        "last_analysis_date": _safe_str(vt_res.get("last_analysis_date"), "N/A"),
        "message": vt_res.get("message"),
        "error": vt_res.get("error"),
    }

    # Support mocked tests omitting explicit `num_distinct_users` vs `distinct_users`
    distinct_val = abuse_res.get("num_distinct_users")
    if distinct_val is None:
        distinct_val = abuse_res.get("distinct_users")

    abuse_norm = {
        "source": "AbuseIPDB",
        "configured": bool(abuse_res.get("configured", bool(config.get_api_key("ABUSEIPDB_API_KEY")) or abuse_res.get("available", False))),
        "status": abuse_res.get("status", "UNKNOWN"),
        "connection_status": abuse_res.get("connection_status") or ("Connected" if abuse_res.get("available") else abuse_res.get("status", "Unavailable")),
        "available": bool(abuse_res.get("available", False)),
        "abuse_confidence_score": max(0, min(100, _safe_int(abuse_res.get("abuse_confidence_score"), 0))),
        "total_reports": max(0, _safe_int(abuse_res.get("total_reports"), 0)),
        "num_distinct_users": max(0, _safe_int(distinct_val, 0)),
        "country": _safe_str(abuse_res.get("country"), "Unknown"),
        "country_name": _safe_str(abuse_res.get("country_name"), _safe_str(abuse_res.get("country"), "Unknown")),
        "isp": _safe_str(abuse_res.get("isp"), "Unknown"),
        "domain": _safe_str(abuse_res.get("domain"), "Unknown"),
        "usage_type": _safe_str(abuse_res.get("usage_type"), "Unknown"),
        "is_tor": bool(abuse_res.get("is_tor", False)),
        "is_whitelisted": bool(abuse_res.get("is_whitelisted", False)),
        "last_reported_at": _safe_str(abuse_res.get("last_reported_at"), "Never"),
        "error": abuse_res.get("error"),
    }

    # Also update runtime connection status when mocked helpers succeed/fail in tests
    _record_provider_state("virustotal", vt_norm["status"], vt_norm["connection_status"], vt_norm["error"])
    _record_provider_state("abuseipdb", abuse_norm["status"], abuse_norm["connection_status"], abuse_norm["error"])

    # Evaluate hybrid risk status
    assessment = evaluate_hybrid_threat_status(ip_meta, vt_norm, abuse_norm)

    # Compile unified geolocation / network metadata (preferring available provider fields over "Unknown")
    def _pick_known(*vals: str, default: str = "Unknown") -> str:
        for v in vals:
            if v and v not in ("Unknown", "N/A", "0"):
                return v
        return default

    country = _pick_known(abuse_norm["country"], vt_norm["country"])
    country_name = _pick_known(abuse_norm["country_name"], country)
    isp = _pick_known(abuse_norm["isp"], vt_norm["as_owner"])
    asn = _pick_known(vt_norm["asn"])
    as_owner = _pick_known(vt_norm["as_owner"], isp)
    domain = _pick_known(abuse_norm["domain"])
    usage_type = _pick_known(abuse_norm["usage_type"])

    result = {
        "success": True,
        "ip_address": ip_meta["ip"],
        "ip_version": ip_meta["version"],
        "ip_type": ip_meta["ip_type"],
        "is_public": True,
        "is_private": False,
        "cached": False,
        "lookup_timestamp": checked_at,
        "validation": ip_meta,
        "country": country,
        "country_name": country_name,
        "isp": isp,
        "domain": domain,
        "asn": asn,
        "as_owner": as_owner,
        "usage_type": usage_type,
        "is_tor": abuse_norm["is_tor"],
        "threat_status": assessment["threat_status"],
        "classification_label": assessment["classification_label"],
        "risk_level": assessment["risk_level"],
        "risk_score": assessment["risk_score"],
        "risk_score_available": assessment["risk_score_available"],
        "reasons": assessment["reasons"],
        "evidence_summary": assessment["evidence_summary"],
        "assessment_summary": assessment["assessment_summary"],
        "recommended_action": assessment["recommended_action"],
        "virustotal": vt_norm,
        "abuseipdb": abuse_norm,
        "api_status": {
            "virustotal": {
                "configured": vt_norm["configured"],
                "status": vt_norm["status"],
                "connection_status": vt_norm["connection_status"],
            },
            "abuseipdb": {
                "configured": abuse_norm["configured"],
                "status": abuse_norm["status"],
                "connection_status": abuse_norm["connection_status"],
            },
        },
    }

    # Cache ONLY when actionable intelligence was returned so transient errors/missing keys are never cached for 24h
    if assessment["risk_score_available"]:
        threat_intel_cache.set(ip_meta["ip"], result)

    return result
