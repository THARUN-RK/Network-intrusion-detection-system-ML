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
