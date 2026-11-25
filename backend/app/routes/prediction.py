"""
FastAPI Routes: Prediction Endpoints (/predict, /predict/batch)
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import asyncio

from backend.app.schemas import (
    FlowFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)
from backend.app.services.model_service import get_model_service, ModelService
from backend.app.websocket_manager import ws_manager

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict_single(flow: FlowFeatures, service: ModelService = Depends(get_model_service)):
    """Classifies a single network flow record using the trained SVM model."""
    try:
        data = flow.model_dump(by_alias=True)
        # Flatten extra features if any
        extra = data.pop("extra_features", {}) or {}
        data.update(extra)
        
        result = service.predict_flow(data)
        
        # Real-time WebSocket alert broadcast
        if result.get("status") == "ATTACK" or result.get("is_anomaly"):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(ws_manager.broadcast_alert(result))
            except Exception:
                pass
                
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest, service: ModelService = Depends(get_model_service)):
    """Classifies a batch of network flow records."""
    try:
        if not request.records:
            raise HTTPException(status_code=400, detail="Empty records list received.")
            
        result = service.predict_batch_flows(request.records)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
