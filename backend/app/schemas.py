"""
Pydantic schemas for FastAPI request and response validation.
Upgraded with SOC Incident Tracking, IP Reputation, and Severity Level Indicators.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class FlowFeatures(BaseModel):
    """Schema for individual network flow feature input."""
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    source_ip: Optional[str] = Field(default="192.168.1.100", alias="Source IP")
    destination_ip: Optional[str] = Field(default="10.0.0.1", alias="Destination IP")
    destination_port: Optional[int] = Field(default=80, alias="Destination Port")
    protocol: Optional[str] = Field(default="TCP", alias="Protocol")
    flow_duration: Optional[float] = Field(default=1000.0, alias="Flow Duration")
    total_fwd_packets: Optional[int] = Field(default=5, alias="Total Fwd Packets")
    total_backward_packets: Optional[int] = Field(default=5, alias="Total Backward Packets")
    flow_bytes_s: Optional[float] = Field(default=1000.0, alias="Flow Bytes/s")
    flow_packets_s: Optional[float] = Field(default=100.0, alias="Flow Packets/s")
    syn_flag_count: Optional[int] = Field(default=0, alias="SYN Flag Count")
    ack_flag_count: Optional[int] = Field(default=1, alias="ACK Flag Count")
    extra_features: Optional[Dict[str, Any]] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    """Schema for single prediction output."""
    record_id: str
    prediction: str
    status: str
    confidence: float
    severity: Optional[str] = "LOW"
    is_anomaly: Optional[bool] = False
    source_ip: Optional[str] = "192.168.1.100"
    destination_ip: Optional[str] = "10.0.0.1"
    protocol: Optional[str] = "TCP"
    probabilities: Optional[Dict[str, float]] = None


class BatchPredictionRequest(BaseModel):
    """Schema for batch network flows input."""
    records: List[Dict[str, Any]]


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction output."""
    total_processed: int
    attacks_detected: int
    results: List[Dict[str, Any]]


class OverviewAnalyticsResponse(BaseModel):
    """Schema for SOC overview statistics."""
    total_records_analysed: int
    total_normal_connections: int
    total_detected_attacks: int
    total_anomalies_detected: Optional[int] = 0
    attack_detection_percentage: float
    most_frequent_attack: str
    category_distribution: Dict[str, int]
    severity_distribution: Optional[Dict[str, int]] = None


class ModelInfoResponse(BaseModel):
    """Schema for model architecture & training metadata."""
    model_type: str
    kernel: str
    best_params: Dict[str, Any]
    feature_count: int
    training_samples_count: int
    classes: List[str]
    cv_macro_f1: Optional[float] = None
    training_duration_seconds: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check status schema."""
    status: str
    model_loaded: bool
    version: str
    simulator_running: bool = False
    last_tick_at: Optional[str] = None
    flows_generated: int = 0
    last_error: Optional[str] = None


# ==============================================================================
# SOC INCIDENT & IP REPUTATION SCHEMAS
# ==============================================================================

class IncidentCreateRequest(BaseModel):
    title: str
    description: str
    severity: str = "HIGH"
    source_ip: Optional[str] = "Unknown"
    target_ip: Optional[str] = "Unknown"
    assigned_to: Optional[str] = "Unassigned"


class IncidentUpdateRequest(BaseModel):
    status: str
    notes: Optional[str] = None


class IncidentResponse(BaseModel):
    id: int
    incident_code: str
    title: str
    description: Optional[str] = None
    severity: str
    status: str
    assigned_to: str
    source_ip: Optional[str] = None
    target_ip: Optional[str] = None
    created_at: str
    updated_at: Optional[str] = None
    notes: Optional[str] = None


class IPReputationResponse(BaseModel):
    ip_address: str
    total_flows: int
    attack_count: int
    reputation_score: float
    threat_level: str
    is_blocked: bool
    country: str
    last_seen: str


class IPBlockRequest(BaseModel):
    ip_address: str
    block: bool


# ==============================================================================
# IP THREAT INTELLIGENCE SCHEMAS
# ==============================================================================

class IPThreatLookupRequest(BaseModel):
    ip_address: str = Field(..., description="Valid public or internal IPv4/IPv6 address")
    force_refresh: bool = Field(default=False, description="Bypass cache and force fresh API query")


class IPThreatLookupResponse(BaseModel):
    ip_address: str
    validation: Dict[str, Any]
    virustotal: Dict[str, Any]
    abuseipdb: Dict[str, Any]
    hybrid_assessment: Dict[str, Any]
    threat_status: str
    risk_score: float
    cached: bool
    checked_at: str


class IPThreatHistoryItem(BaseModel):
    id: int
    ip_address: str
    ip_version: int
    is_public: bool
    threat_status: str
    risk_score: float
    abuse_confidence: float
    vt_detection_ratio: str
    total_reports: int
    country: str
    isp: str
    asn: str
    threat_reasons: List[str] = Field(default_factory=list)
    raw_intel: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str


class IPThreatHistoryResponse(BaseModel):
    total: int
    items: List[IPThreatHistoryItem]


class IPThreatKpiResponse(BaseModel):
    total_lookups: int
    malicious_count: int
    suspicious_count: int
    clean_count: int
    unknown_count: int
    top_countries: List[Dict[str, Any]] = Field(default_factory=list)


class UnifiedDashboardStatsResponse(BaseModel):
    total_records_analysed: int
    total_normal_connections: int
    total_detected_attacks: int
    attack_detection_percentage: float
    total_ips_checked: int
    malicious_ips: int
    suspicious_ips: int
    clean_ips: int
    category_distribution: Dict[str, int]
    threat_status_distribution: Dict[str, int]


# ==============================================================================
# IP-BASED NETWORK ATTACK ANALYZER SCHEMAS
# ==============================================================================

class IPAttackAnalyzerRequest(BaseModel):
    ip_address: str = Field(..., description="Source IPv4 or IPv6 address to analyze")


class FlowAnalysisItem(BaseModel):
    flow_index: int
    record_id: str
    timestamp: str
    destination_port: int
    flow_duration: float
    predicted_class: str
    security_status: str
    confidence: float
    severity: str
    probabilities: Optional[Dict[str, float]] = None


class IPAttackAnalyzerResponse(BaseModel):
    has_data: bool
    status: str
    message: Optional[str] = None
    source_ip: str
    validation: Optional[Dict[str, Any]] = None
    total_flows: int = 0
    benign_flow_count: int = 0
    malicious_flow_count: int = 0
    dos_flow_count: int = 0
    ddos_flow_count: int = 0
    portscan_flow_count: int = 0
    brute_force_flow_count: int = 0
    other_attacks_count: int = 0
    most_frequent_attack: Optional[str] = None
    average_confidence: Optional[float] = None
    attack_severity: Optional[str] = None
    class_distribution: Optional[Dict[str, int]] = None
    flow_records: List[FlowAnalysisItem] = Field(default_factory=list)
    distinction_note: Optional[str] = None
