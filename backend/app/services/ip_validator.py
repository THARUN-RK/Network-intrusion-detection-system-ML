"""
IP Address Validation and Classification Service
Supports IPv4 and IPv6 validation, public/private/reserved categorization,
and threat intelligence eligibility checks according to RFC standards.
"""

import ipaddress
from typing import Dict, Any, Optional, Tuple


# RFC 6598 Shared Address Space (Carrier-Grade NAT)
CGNAT_NETWORK = ipaddress.IPv4Network("100.64.0.0/10")

# Documentation networks (RFC 5737 and RFC 3849)
DOC_NETWORKS = [
    ipaddress.IPv4Network("192.0.2.0/24"),
    ipaddress.IPv4Network("198.51.100.0/24"),
    ipaddress.IPv4Network("203.0.113.0/24"),
    ipaddress.IPv6Network("2001:db8::/32")
]


def validate_and_classify_ip(ip_input: str) -> Dict[str, Any]:
    """
    Validates any IPv4 or IPv6 address string and classifies its scope.
    
    Returns a standardized dictionary containing:
    - is_valid: bool
    - ip: str (normalized)
    - version: Optional[int] (4 or 6)
    - ip_type: str (PUBLIC, PRIVATE, LOOPBACK, LINK_LOCAL, MULTICAST, RESERVED, CGNAT, DOCUMENTATION)
    - is_public: bool
    - is_private: bool
    - lookup_eligible: bool (True only if public routable IP eligible for external threat feeds)
    - error: Optional[str]
    - message: str
    """
    if not ip_input or not isinstance(ip_input, str):
        return {
            "is_valid": False,
            "ip": str(ip_input or ""),
            "version": None,
            "ip_type": "INVALID",
            "is_public": False,
            "is_private": False,
            "lookup_eligible": False,
            "error": "IP address cannot be empty.",
            "message": "Invalid input: empty string provided."
        }

    clean_ip = ip_input.strip()

    # Reject URLs, CIDR blocks, or ports passed incorrectly
    if "/" in clean_ip:
        return {
            "is_valid": False,
            "ip": clean_ip,
            "version": None,
            "ip_type": "INVALID",
            "is_public": False,
            "is_private": False,
            "lookup_eligible": False,
            "error": "CIDR prefix notation (e.g. /24) is not supported. Please provide a single host IP.",
            "message": "CIDR notation detected. Expected a single host IP address."
        }

    if ":" in clean_ip and not ("[" in clean_ip or clean_ip.count(":") >= 2):
        # Likely an IPv4 address with port (e.g. 1.2.3.4:8080)
        return {
            "is_valid": False,
            "ip": clean_ip,
            "version": None,
            "ip_type": "INVALID",
            "is_public": False,
            "is_private": False,
            "lookup_eligible": False,
            "error": "Port numbers are not supported in IP reputation lookups. Please provide the bare IP.",
            "message": "Port notation detected. Strip the port number and try again."
        }

    try:
        obj = ipaddress.ip_address(clean_ip)
    except ValueError as e:
        return {
            "is_valid": False,
            "ip": clean_ip,
            "version": None,
            "ip_type": "INVALID",
            "is_public": False,
            "is_private": False,
            "lookup_eligible": False,
            "error": f"Invalid IP address format: {str(e)}",
            "message": "Provided string is neither a valid IPv4 nor a valid IPv6 address."
        }

    version = obj.version
    normalized_ip = str(obj)

    # Classification logic
    if obj.is_loopback:
        ip_type = "LOOPBACK"
        message = f"Loopback address ({normalized_ip}) refers to the local host machine."
    elif obj.is_link_local:
        ip_type = "LINK_LOCAL"
        message = f"Link-local address ({normalized_ip}) is only valid for communication within the local network segment."
    elif obj.is_multicast:
        ip_type = "MULTICAST"
        message = f"Multicast address ({normalized_ip}) used for one-to-many communication."
    elif obj.is_unspecified:
        ip_type = "RESERVED"
        message = f"Unspecified address ({normalized_ip}) representing all local interfaces."
    elif any(obj in net for net in DOC_NETWORKS):
        ip_type = "DOCUMENTATION"
        message = f"Documentation range address ({normalized_ip}) reserved for examples and documentation."
    elif version == 4 and obj in CGNAT_NETWORK:
        ip_type = "CGNAT"
        message = f"Carrier-Grade NAT address ({normalized_ip}) reserved for ISP shared address space (RFC 6598)."
    elif obj.is_private:
        ip_type = "PRIVATE"
        message = f"Private RFC {'1918' if version == 4 else '4193'} address ({normalized_ip}) used inside internal local networks."
    elif obj.is_reserved:
        ip_type = "RESERVED"
        message = f"IETF reserved address ({normalized_ip}) for future or special protocol use."
    elif obj.is_global:
        ip_type = "PUBLIC"
        message = f"Public globally-routable IPv{version} address ({normalized_ip})."
    else:
        ip_type = "NON_GLOBAL"
        message = f"Non-global IPv{version} address ({normalized_ip})."

    is_public = (ip_type == "PUBLIC")
    is_private = (ip_type in ["PRIVATE", "LOOPBACK", "LINK_LOCAL"])
    lookup_eligible = is_public

    return {
        "is_valid": True,
        "ip": normalized_ip,
        "version": version,
        "ip_type": ip_type,
        "is_public": is_public,
        "is_private": is_private,
        "lookup_eligible": lookup_eligible,
        "error": None,
        "message": message
    }
