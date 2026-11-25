"""
Real-Time Event Stream Manager (SSE & WebSocket) for NIDS SOC
Broadcasts live predictions, security alerts, and statistical updates to connected browser clients.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Set, Dict, Any

logger = logging.getLogger(__name__)


class StreamBroadcaster:
    """Thread-safe event broadcaster supporting Server-Sent Events (SSE) and WebSockets."""

    def __init__(self):
        self._listeners: Set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        """Subscribes a new client queue with bounded capacity."""
        q: asyncio.Queue = asyncio.Queue(maxsize=150)
        self._listeners.add(q)
        logger.debug(f"[SSE Stream] Client subscribed. Total listeners: {len(self._listeners)}")
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Removes a disconnected client queue."""
        self._listeners.discard(q)
        logger.debug(f"[SSE Stream] Client unsubscribed. Total listeners: {len(self._listeners)}")

    async def broadcast(self, event_type: str, data: Dict[str, Any]) -> None:
        """Dispatches an event payload to all active listener queues."""
        if not self._listeners:
            return

        payload = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Dispatch non-blocking; drop if queue is saturated
        dead_queues = set()
        for q in list(self._listeners):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass
            except Exception:
                dead_queues.add(q)

        for dead in dead_queues:
            self._listeners.discard(dead)


# Global singleton instance
stream_broadcaster = StreamBroadcaster()
