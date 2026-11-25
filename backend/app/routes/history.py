"""
FastAPI Routes: Prediction History & Incident Alerts
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import database
from backend.app.simulator import traffic_simulator

router = APIRouter(tags=["History & Alerts"])


@router.get("/history")
def get_history(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = None,
    category: Optional[str] = None
):
    """Retrieves paginated prediction audit trail."""
    try:
        limit_val = limit if isinstance(limit, int) else 50
        offset_val = offset if isinstance(offset, int) else 0
        status_val = status if isinstance(status, str) else None
        category_val = category if isinstance(category, str) else None

        records = database.get_prediction_history(
            limit=limit_val,
            offset=offset_val,
            status_filter=status_val,
            category_filter=category_val
        )
        return {"total_returned": len(records), "records": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
def get_alerts(
    limit: int = Query(default=20, ge=1, le=100),
    since_id: Optional[int] = Query(default=None, ge=0)
):
    """Retrieves recent security incident alerts, with optional since_id polling filter."""
    try:
        traffic_simulator.catch_up_sync()
        limit_val = limit if isinstance(limit, int) else 20
        since_id_val = since_id if isinstance(since_id, int) else None
        alerts = database.get_recent_alerts(limit=limit_val, since_id=since_id_val)
        return {"total_alerts": len(alerts), "alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
