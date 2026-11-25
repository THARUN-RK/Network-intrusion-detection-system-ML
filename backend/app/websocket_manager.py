"""
Network Intrusion Detection System (NIDS) - WebSocket Connection Manager
Broadcasts live security alerts, traffic metrics, and threat pings to active SOC dashboards.
"""

import json
import logging
from typing import List
from fastapi import WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket clients and handles broadcast dispatch."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Active clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Remaining: {len(self.active_connections)}")

    async def broadcast_alert(self, alert_data: dict):
        """Dispatches an intrusion alert notification to all connected dashboards."""
        if not self.active_connections:
            return
            
        payload = json.dumps({"type": "SECURITY_ALERT", "data": alert_data})
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                disconnected.append(connection)
                
        for dead in disconnected:
            self.disconnect(dead)

    async def broadcast_traffic_metric(self, metric_data: dict):
        """Pushes live packet/flow metrics."""
        if not self.active_connections:
            return
            
        payload = json.dumps({"type": "TRAFFIC_UPDATE", "data": metric_data})
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                disconnected.append(connection)
                
        for dead in disconnected:
            self.disconnect(dead)


# Global singleton instance
ws_manager = ConnectionManager()
