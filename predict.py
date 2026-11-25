"""
Network Intrusion Detection System (NIDS) - Prediction Engine
Loads the trained multiclass SVM pipeline, validates incoming network features,
computes class probabilities, and determines security status.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

import joblib
import numpy as np
import pandas as pd

import config
import database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Confidence threshold below which traffic is flagged as an unknown anomaly (Zero-Day defense)
ANOMALY_CONFIDENCE_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "55.0"))


class NIDSPredictor:
    """
    Production-grade inference engine for network flow attack classification.
    Supports unknown traffic rejection, confidence calibration, and IP tracking.
    """
    def __init__(
        self,
        model_path: Path = config.MODEL_PATH,
        metadata_path: Path = config.METADATA_PATH
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.pipeline = None
        self.metadata = {}
        self.expected_features: List[str] = []
        self.classes: List[str] = []
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads serialized model pipeline and metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}. Train the model first.")
            
        logger.info(f"Loading SVM model from {self.model_path}...")
        self.pipeline = joblib.load(self.model_path)
        
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.expected_features = self.metadata.get("feature_names", config.CORE_FLOW_FEATURES)
            self.classes = self.metadata.get("classes", list(self.pipeline.classes_))
        else:
            self.expected_features = config.CORE_FLOW_FEATURES
            self.classes = list(self.pipeline.classes_)
            
        logger.info(f"Model loaded successfully with {len(self.expected_features)} features and {len(self.classes)} classes.")

    def validate_and_align_features(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures the input DataFrame contains exactly the features expected by the trained model
        in identical order. Fills missing features with median/zero, sanitizes non-numeric types.
        """
        df = df_input.copy()
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        # Add any missing expected columns with 0.0
        for feat in self.expected_features:
            if feat not in df.columns:
                df[feat] = 0.0
                
        # Reorder columns to exact schema
        df_aligned = df[self.expected_features].copy()
        
        # Convert all to float
        for col in df_aligned.columns:
            df_aligned[col] = pd.to_numeric(df_aligned[col], errors="coerce")
            
        return df_aligned

    def predict_single_flow(
        self,
        record: Dict[str, Any],
        record_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        destination_ip: Optional[str] = None,
        protocol: Optional[str] = None,
        log_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        Classifies a single network flow record with confidence calibration and unknown traffic rejection.
        """
        df_single = pd.DataFrame([record])
        df_aligned = self.validate_and_align_features(df_single)
        
        # Predict class
        predicted_class = self.pipeline.predict(df_aligned)[0]
        
        # Compute confidence from predict_proba
        confidence = 0.0
        max_confidence = 0.0
        prob_dict = {}
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(df_aligned)[0]
            class_indices = {cls: idx for idx, cls in enumerate(self.pipeline.classes_)}
            pred_idx = class_indices.get(predicted_class, 0)
            confidence = round(float(probs[pred_idx]) * 100.0, 2)
            max_confidence = round(float(np.max(probs)) * 100.0, 2)
            prob_dict = {cls: round(float(p) * 100.0, 2) for cls, p in zip(self.pipeline.classes_, probs)}
        else:
            confidence = 100.0
            max_confidence = 100.0
            
        # Unknown Traffic Rejection (Zero-day defense): reject ANY prediction below threshold
        is_anomaly = False
        if max_confidence < ANOMALY_CONFIDENCE_THRESHOLD:
            is_anomaly = True
            predicted_class = "ANOMALY (Unknown Threat)"
            security_status = "ATTACK"
            confidence = max_confidence
        else:
            security_status = "BENIGN" if predicted_class == "BENIGN" else "ATTACK"
            
        src_ip = source_ip or record.get("source_ip", record.get("Source IP", "192.168.1.100"))
        dst_ip = destination_ip or record.get("destination_ip", record.get("Destination IP", "10.0.0.1"))
        proto = protocol or record.get("protocol", record.get("Protocol", "TCP"))
        sev = database.map_severity(predicted_class, confidence, is_anomaly)
        
        result = {
            "prediction": predicted_class,
            "status": security_status,
            "confidence": confidence,
            "severity": sev,
            "is_anomaly": is_anomaly,
            "source_ip": src_ip,
            "destination_ip": dst_ip,
            "protocol": proto,
            "probabilities": prob_dict,
            "record_id": record_id or f"flow-{int(pd.Timestamp.now().timestamp()*1000)}"
        }
        
        if log_to_db:
            database.save_prediction(
                predicted_category=predicted_class,
                confidence=confidence,
                security_status=security_status,
                record_identifier=result["record_id"],
                destination_port=int(record.get("Destination Port", 80)) if "Destination Port" in record else 80,
                flow_duration=float(record.get("Flow Duration", 0.0)) if "Flow Duration" in record else 0.0,
                source_ip=src_ip,
                destination_ip=dst_ip,
                protocol=proto,
                is_anomaly=is_anomaly,
                metadata={"probabilities": prob_dict}
            )
            
        return result

    def predict_batch(
        self,
        df_batch: pd.DataFrame,
        log_to_db: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Classifies a DataFrame of network flows.
        """
        if df_batch.empty:
            return []
            
        df_aligned = self.validate_and_align_features(df_batch)
        
        preds = self.pipeline.predict(df_aligned)
        
        has_proba = hasattr(self.pipeline, "predict_proba")
        probs_matrix = self.pipeline.predict_proba(df_aligned) if has_proba else None
        
        results = []
        db_records = []
        
        for i, pred_class in enumerate(preds):
            if has_proba and probs_matrix is not None:
                class_idx = list(self.pipeline.classes_).index(pred_class)
                conf = round(float(probs_matrix[i, class_idx]) * 100.0, 2)
                max_conf = round(float(np.max(probs_matrix[i])) * 100.0, 2)
            else:
                conf = 100.0
                max_conf = 100.0
                
            is_ano = False
            if max_conf < ANOMALY_CONFIDENCE_THRESHOLD:
                is_ano = True
                pred_class = "ANOMALY (Unknown Threat)"
                sec_status = "ATTACK"
                conf = max_conf
            else:
                sec_status = "BENIGN" if pred_class == "BENIGN" else "ATTACK"
                
            rec_id = f"batch-flow-{i+1}"
            src_ip = str(df_batch.iloc[i].get("Source IP", df_batch.iloc[i].get("source_ip", "192.168.1.100")))
            dst_ip = str(df_batch.iloc[i].get("Destination IP", df_batch.iloc[i].get("destination_ip", "10.0.0.1")))
            
            res_item = {
                "record_id": rec_id,
                "prediction": pred_class,
                "status": sec_status,
                "confidence": conf,
                "is_anomaly": is_ano,
                "source_ip": src_ip,
                "destination_ip": dst_ip
            }
            results.append(res_item)
            
            if log_to_db:
                db_records.append({
                    "record_identifier": rec_id,
                    "predicted_category": pred_class,
                    "confidence": conf,
                    "security_status": sec_status,
                    "is_anomaly": is_ano,
                    "source_ip": src_ip,
                    "destination_ip": dst_ip,
                    "destination_port": int(df_batch.iloc[i].get("Destination Port", 80)) if "Destination Port" in df_batch.columns else 80,
                    "flow_duration": float(df_batch.iloc[i].get("Flow Duration", 0.0)) if "Flow Duration" in df_batch.columns else 0.0
                })
                
        if log_to_db and db_records:
            database.save_predictions_batch(db_records)
            
        return results


# Module-level singleton
_predictor_instance = None

def get_predictor() -> NIDSPredictor:
    """Returns singleton instance of the predictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = NIDSPredictor()
    return _predictor_instance


if __name__ == "__main__":
    predictor = get_predictor()
    # Test single flow inference
    sample_flow = {
        "Destination Port": 80,
        "Flow Duration": 15000.0,
        "Total Fwd Packets": 100,
        "Total Backward Packets": 2,
        "Flow Bytes/s": 3500000.0,
        "Flow Packets/s": 45000.0,
        "SYN Flag Count": 1,
        "ACK Flag Count": 0
    }
    result = predictor.predict_single_flow(sample_flow)
    print("Inference Test Result:\n", json.dumps(result, indent=2))
