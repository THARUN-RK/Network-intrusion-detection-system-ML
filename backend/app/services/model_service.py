"""
FastAPI Model Service - Wraps predictor and metadata for endpoint consumers.
"""

from typing import Dict, Any, List
import pandas as pd
import predict
import config


class ModelService:
    def __init__(self):
        self.predictor = predict.get_predictor()

    def predict_flow(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return self.predictor.predict_single_flow(record, log_to_db=True)

    def predict_batch_flows(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        df = pd.DataFrame(records)
        results = self.predictor.predict_batch(df, log_to_db=True)
        attacks_count = sum(1 for r in results if r["status"] == "ATTACK")
        return {
            "total_processed": len(results),
            "attacks_detected": attacks_count,
            "results": results
        }

    def get_model_metadata(self) -> Dict[str, Any]:
        return self.predictor.metadata


_model_service_instance = None

def get_model_service() -> ModelService:
    global _model_service_instance
    if _model_service_instance is None:
        _model_service_instance = ModelService()
    return _model_service_instance
