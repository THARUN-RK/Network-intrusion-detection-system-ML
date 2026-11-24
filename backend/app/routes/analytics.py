"""
FastAPI Routes: Analytics & Insights Endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import database
from backend.app.schemas import OverviewAnalyticsResponse

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/overview", response_model=OverviewAnalyticsResponse)
def get_overview_analytics():
    """Returns SOC summary statistics on traffic and attack rates."""
    try:
        stats = database.get_overview_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")


@router.get("/attacks")
def get_attack_distribution():
    """Returns detailed attack counts by category."""
    try:
        stats = database.get_overview_statistics()
        return {"category_distribution": stats["category_distribution"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/daily")
def get_daily_trends():
    """Returns timestamped detection trends."""
    try:
        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT strftime('%Y-%m-%d %H:00:00', timestamp) as hour_bucket,
                   security_status,
                   COUNT(*) as count
            FROM predictions
            GROUP BY hour_bucket, security_status
            ORDER BY hour_bucket DESC
            LIMIT 48
        """)
        rows = cursor.fetchall()
        conn.close()
        
        trends = [{"hour": r[0], "status": r[1], "count": r[2]} for r in rows]
        return {"hourly_trends": trends}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
