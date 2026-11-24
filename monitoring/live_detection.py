"""
Network Intrusion Detection System (NIDS) - Live Threat Detection Orchestrator
Coordinates live packet capture, flow feature extraction, SVM prediction, and alert generation.
"""

import sys
import logging
from typing import Dict, Any

import predict
import database
from monitoring.packet_capture import start_live_capture

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def handle_completed_flow(flow_features: Dict[str, Any]) -> None:
    """Callback invoked when a network flow is closed or timed out."""
    predictor = predict.get_predictor()
    result = predictor.predict_single_flow(flow_features, log_to_db=True)
    
    pred_class = result["prediction"]
    status = result["status"]
    conf = result["confidence"]
    
    if status == "ATTACK":
        logger.warning(f"🚨 [ALERT] INTRUSION DETECTED: {pred_class} (Confidence: {conf:.1f}%)")
    else:
        logger.info(f"✓ [BENIGN] Normal Flow (Confidence: {conf:.1f}%)")


def run_live_monitoring(interface: str = None, count: int = 50):
    """Executes live traffic detection."""
    logger.info("Initializing Live Network Intrusion Detection Engine...")
    start_live_capture(interface=interface, packet_count=count, callback=handle_completed_flow)


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run_live_monitoring(count=count)
