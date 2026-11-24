"""
Pydantic schemas for FastAPI request and response validation.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class FlowFeatures(BaseModel):
    """Schema for individual network flow feature input."""
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    destination_port: Optional[int] = Field(default=80, alias="Destination Port")
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
    attack_detection_percentage: float
    most_frequent_attack: str
    category_distribution: Dict[str, int]


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
