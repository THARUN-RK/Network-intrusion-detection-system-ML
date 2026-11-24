"""
Unit tests for live monitoring and flow feature extraction.
"""

import pytest
from monitoring.flow_extractor import NetworkFlow
import predict


def test_network_flow_feature_extraction():
    key = ("192.168.1.10", "10.0.0.1", 54321, 80, "TCP")
    flow = NetworkFlow(key, start_time=100.0)
    
    # Forward packet (SYN)
    flow.add_packet(length=54, timestamp=100.0, is_forward=True, flags={"SYN": 1, "ACK": 0})
    # Backward packet (SYN-ACK)
    flow.add_packet(length=60, timestamp=100.02, is_forward=False, flags={"SYN": 1, "ACK": 1})
    # Forward packet (ACK + Data)
    flow.add_packet(length=500, timestamp=100.05, is_forward=True, flags={"SYN": 0, "ACK": 1, "PSH": 1})
    
    features = flow.extract_features()
    
    assert features["Destination Port"] == 80
    assert features["Total Fwd Packets"] == 2
    assert features["Total Backward Packets"] == 1
    assert features["Total Length of Fwd Packets"] == 554
    assert features["Total Length of Bwd Packets"] == 60
    assert features["SYN Flag Count"] == 2
    assert features["ACK Flag Count"] == 2
    assert features["PSH Flag Count"] == 1
    assert features["Flow Duration"] > 0
    assert len(features) == 71


def test_flow_features_compatibility_with_model():
    key = ("10.0.0.5", "192.168.1.1", 45000, 443, "TCP")
    flow = NetworkFlow(key, start_time=200.0)
    flow.add_packet(100, 200.1, True, {"SYN": 1})
    flow.add_packet(200, 200.2, False, {"ACK": 1})
    
    features = flow.extract_features()
    predictor = predict.get_predictor()
    result = predictor.predict_single_flow(features, log_to_db=False)
    
    assert "prediction" in result
    assert result["status"] in ["BENIGN", "ATTACK"]
    assert 0.0 <= result["confidence"] <= 100.0
