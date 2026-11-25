"""
FastAPI Routes: Real-Time SOC Live Endpoints (/api/stats, /api/timeseries, /api/stream)
"""

import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Request, Query, HTTPException
from fastapi.responses import StreamingResponse

import database
from backend.app.stream_manager import stream_broadcaster
from backend.app import simulator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Live Real-Time SOC"])


@router.get("/stats")
def get_live_soc_stats(window_minutes: int = Query(5, ge=1, le=60)):
    """
    Returns real-time aggregated metrics: total flows, benign count, attack count,
    detection percentage, flows per minute, trend, and attack class counts in the last 5 minutes.
    """
    try:
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
