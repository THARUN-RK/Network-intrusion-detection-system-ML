"""
FastAPI Routes: Real-Time SOC Live Endpoints (/api/stats, /api/timeseries, /api/stream)
"""

import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import APIRouter, Request, Query, HTTPException
from fastapi.responses import StreamingResponse, FileResponse

import database
from backend.app.stream_manager import stream_broadcaster
from backend.app import simulator
import predict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Live Real-Time SOC"])


@router.get("/stats")
def get_live_soc_stats(window_minutes: int = Query(5, ge=1, le=60)):
    """
    Returns real-time aggregated metrics: total flows, benign count, attack count,
    detection percentage, flows per minute, trend, and attack class counts in the last 5 minutes.
    """
    try:
        simulator.traffic_simulator.catch_up_sync()
        return database.get_live_stats(window_minutes=window_minutes)
    except Exception as e:
        logger.error(f"Error fetching live stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timeseries")
def get_live_soc_timeseries(window_minutes: int = Query(30, ge=5, le=120)):
    """
    Returns minute-by-minute breakdown of total, benign, and attack flows for the last 30 minutes.
    """
    try:
        simulator.traffic_simulator.catch_up_sync()
        return database.get_timeseries_stats(window_minutes=window_minutes)
    except Exception as e:
        logger.error(f"Error fetching timeseries stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream")
async def sse_event_stream(request: Request):
    """
    Server-Sent Events (SSE) stream broadcasting live flow predictions and threat alerts.
    Gracefully disconnects when the client closes the connection.
    """
    async def event_generator():
        client_queue = stream_broadcaster.subscribe()
        try:
            # Send immediate connection handshake
            yield f"event: connected\ndata: {json.dumps({'status': 'connected', 'simulator': simulator.SIMULATOR_ENABLED})}\n\n"
            
            while True:
                if await request.is_disconnected():
                    break
                try:
                    # Wait up to 10 seconds for next event, else send keepalive heartbeat
                    msg = await asyncio.wait_for(client_queue.get(), timeout=10.0)
                    yield f"event: {msg['type']}\ndata: {json.dumps(msg['data'])}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            stream_broadcaster.unsubscribe(client_queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/simulator/toggle")
def toggle_simulator(enabled: Optional[bool] = None):
    """Dynamically enables or disables the background traffic simulator."""
    if enabled is None:
        simulator.SIMULATOR_ENABLED = not simulator.SIMULATOR_ENABLED
    else:
        simulator.SIMULATOR_ENABLED = enabled
    return {"simulator_enabled": simulator.SIMULATOR_ENABLED}


@router.get("/model-metrics")
def get_model_evaluation_metrics():
    """
    Returns authentic, verified evaluation metrics read directly from reports/evaluation_report.json
    or results/metrics.json. Never returns fabricated or invented values.
    """
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    report_file = root_dir / "reports" / "evaluation_report.json"
    results_file = root_dir / "results" / "metrics.json"

    metrics_data = None
    source_file = None

    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
                source_file = "reports/evaluation_report.json"
        except Exception as e:
            logger.warning(f"Failed to parse {report_file}: {e}")

    if not metrics_data and results_file.exists():
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
                source_file = "results/metrics.json"
        except Exception as e:
            logger.warning(f"Failed to parse {results_file}: {e}")

    if not metrics_data:
        return {
            "available": False,
            "message": "Evaluation results not available"
        }

    predictor = predict.get_predictor()
    meta = predictor.metadata
    accuracy = metrics_data.get("accuracy")
    error_percentage = round((1.0 - float(accuracy)) * 100.0, 2) if accuracy is not None else None

    return {
        "available": True,
        "source_file": source_file,
        "model_type": meta.get("model_type", "Support Vector Machine"),
        "kernel": meta.get("kernel", "rbf"),
        "dataset": "CIC-IDS2017 Benchmark",
        "feature_count": meta.get("feature_count", len(predictor.expected_features)),
        "accuracy": accuracy,
        "error_percentage": error_percentage,
        "macro_precision": metrics_data.get("macro_precision"),
        "macro_recall": metrics_data.get("macro_recall"),
        "macro_f1_score": metrics_data.get("macro_f1_score"),
        "weighted_f1_score": metrics_data.get("weighted_f1_score"),
        "macro_pr_auc": metrics_data.get("macro_pr_auc"),
        "prediction_latency_ms": metrics_data.get("prediction_latency_ms_per_record"),
        "total_test_samples": metrics_data.get("total_test_samples"),
        "classes": metrics_data.get("classes", predictor.classes),
        "per_class_metrics": metrics_data.get("per_class_metrics", {})
    }


@router.get("/model-report")
def download_model_report():
    """Returns the genuine evaluation report JSON file."""
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    report_file = root_dir / "reports" / "evaluation_report.json"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Evaluation results not available")
    return FileResponse(report_file, media_type="application/json", filename="evaluation_report.json")

