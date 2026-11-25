"""
Real-Time Background Network Traffic Simulator for NIDS SOC
Samples genuine CIC-IDS2017 benchmark flows, introduces subtle variance,
executes model inference via the standard prediction pipeline, and updates database records.
"""

import os
import time
import random
import asyncio
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

import config
import predict
from backend.app.stream_manager import stream_broadcaster

logger = logging.getLogger(__name__)

# Simulator Configuration from Environment
SIMULATOR_ENABLED = os.getenv("SIMULATOR_ENABLED", "true").lower() in ["true", "1", "yes"]
SIMULATOR_RATE = float(os.getenv("SIMULATOR_RATE", "1.5"))

BENIGN_IPS = [
    "192.168.1.101", "192.168.1.102", "192.168.1.105", "192.168.1.108",
    "192.168.1.115", "10.0.0.15", "10.0.0.22", "172.16.1.50"
]

TARGET_SERVERS = [
    "10.0.0.1", "10.0.0.2", "10.0.0.5", "192.168.1.1", "172.16.0.10"
]

ATTACK_IPS = [
    "185.220.101.5", "194.26.29.112", "45.33.32.156", "103.251.167.20",
    "198.51.100.42", "203.0.113.88", "172.16.0.4", "198.51.100.15",
    "91.240.118.22", "80.82.77.33"
]


class NetworkTrafficSimulator:
    """Simulates realistic production network traffic with attack bursts and real model inference."""

    def __init__(self, sample_path: Optional[Path] = None):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.custom_sample_path = sample_path
        self.active_sample_path: Optional[Path] = None
        self.benign_pool: List[Dict[str, Any]] = []
        self.attack_pool: Dict[str, List[Dict[str, Any]]] = {}
        self.running: bool = False
        self._task: Optional[asyncio.Task] = None
        
        # State & Diagnostics
        self.last_tick_time: float = time.time()
        self.last_tick_at: Optional[str] = None
        self.flows_generated: int = 0
        self.last_error: Optional[str] = None
        
        # Burst State
        self._burst_counter: int = 0
        self._active_burst_category: Optional[str] = None
        self._active_burst_attacker_ip: Optional[str] = None
        self._burst_remaining: int = 0

        self._load_pools()

    def _resolve_sample_path(self) -> Optional[Path]:
        """Resolves sample dataset path relative to codebase root with fallback."""
        if self.custom_sample_path and self.custom_sample_path.exists():
            return self.custom_sample_path

        primary_path = self.project_root / "dataset" / "raw" / "CICIDS2017_Benchmark_Sample.csv"
        if primary_path.exists():
            return primary_path

        fallback_path = self.project_root / "data" / "sample_flows.csv"
        if fallback_path.exists():
            return fallback_path

        return None

    def _load_pools(self) -> None:
        """Loads and indexes sample flows from CIC-IDS2017 held-out benchmark or committed sample."""
        resolved = self._resolve_sample_path()
        if not resolved:
            err_msg = (
                f"No sample file found at primary ({self.project_root / 'dataset' / 'raw' / 'CICIDS2017_Benchmark_Sample.csv'}) "
                f"or fallback ({self.project_root / 'data' / 'sample_flows.csv'}). Simulator cannot load pools."
            )
            self.last_error = err_msg
            logger.error(f"[Traffic Simulator] {err_msg}")
            return

        self.active_sample_path = resolved
        try:
            df = pd.read_csv(resolved, nrows=3000)
            label_col = [c for c in df.columns if "label" in c.lower()]
            if not label_col:
                err_msg = f"Missing label column in sample file {resolved}."
                self.last_error = err_msg
                logger.error(f"[Traffic Simulator] {err_msg}")
                return

            label_col_name = label_col[0]

            self.benign_pool = []
            self.attack_pool = {}

            for _, row in df.iterrows():
                row_dict = row.to_dict()
                label = str(row_dict.pop(label_col_name, "BENIGN")).strip()

                # Clean numeric values
                clean_row = {}
                for k, v in row_dict.items():
                    try:
                        clean_row[k] = float(v) if pd.notnull(v) else 0.0
                    except (ValueError, TypeError):
                        clean_row[k] = 0.0

                if label == "BENIGN":
                    self.benign_pool.append(clean_row)
                else:
                    if label not in self.attack_pool:
                        self.attack_pool[label] = []
                    self.attack_pool[label].append(clean_row)

            total_attacks = sum(len(v) for v in self.attack_pool.values())
            logger.info(
                f"[Traffic Simulator] Loaded pools from {resolved.name}: {len(self.benign_pool)} benign flows, "
                f"{total_attacks} attacks across {len(self.attack_pool)} categories."
            )
            self.last_error = None
        except Exception as e:
            self.last_error = f"Failed to load sample pools: {e}"
            logger.exception(f"[Traffic Simulator] Failed to load sample pools from {resolved}: {e}")

    def _get_next_flow(self) -> Tuple[Dict[str, Any], str, str]:
        """
        Picks the next flow vector, source IP, and target IP based on realistic distribution
        or ongoing attack burst.
        """
        # Lazy retry pool load if pools were empty
        if not self.benign_pool and not self.attack_pool:
            self._load_pools()

        if not self.benign_pool and not self.attack_pool:
            return {}, "192.168.1.100", "10.0.0.1"

        # Handle active burst
        if self._burst_remaining > 0 and self._active_burst_category and self._active_burst_attacker_ip:
            self._burst_remaining -= 1
            pool = self.attack_pool.get(self._active_burst_category)
            if pool:
                flow = dict(random.choice(pool))
            else:
                flow = dict(random.choice(self.benign_pool)) if self.benign_pool else {}
            src_ip = self._active_burst_attacker_ip
            dst_ip = random.choice(TARGET_SERVERS)
            return flow, src_ip, dst_ip

        # Check if we should trigger a new burst (every ~30 normal cycles)
        self._burst_counter += 1
        if self._burst_counter >= random.randint(25, 45) and self.attack_pool:
            self._burst_counter = 0
            burst_cat = random.choice(["DDoS", "PortScan", "DoS"] if "DDoS" in self.attack_pool else list(self.attack_pool.keys()))
            self._active_burst_category = burst_cat
            self._active_burst_attacker_ip = random.choice(ATTACK_IPS)
            self._burst_remaining = random.randint(10, 25)
            logger.info(f"[Traffic Simulator] Triggering {burst_cat} attack burst of {self._burst_remaining} flows from {self._active_burst_attacker_ip}!")
            return self._get_next_flow()

        # Normal realistic mix: ~88% Benign, ~12% occasional single attack
        is_attack = (random.random() < 0.12) and bool(self.attack_pool)
        if is_attack:
            cat = random.choice(list(self.attack_pool.keys()))
            flow = dict(random.choice(self.attack_pool[cat]))
            src_ip = random.choice(ATTACK_IPS)
            dst_ip = random.choice(TARGET_SERVERS)
        else:
            flow = dict(random.choice(self.benign_pool)) if self.benign_pool else {}
            src_ip = random.choice(BENIGN_IPS)
            dst_ip = random.choice(TARGET_SERVERS)

        return flow, src_ip, dst_ip

    def _apply_subtle_noise(self, flow: Dict[str, Any]) -> Dict[str, Any]:
        """Applies gentle numeric variance to keep flow metrics dynamic and realistic."""
        noised = {}
        for k, v in flow.items():
            if k in ["Destination Port", "Protocol"] or not isinstance(v, (int, float)):
                noised[k] = v
            else:
                factor = 1.0 + random.uniform(-0.02, 0.02)
                noised[k] = max(0.0, float(v) * factor)
        return noised

    def _predict_and_store_sync(self, predictor: predict.NIDSPredictor, flow: Dict[str, Any], src_ip: str, dst_ip: str) -> Dict[str, Any]:
        """Synchronous wrapper for prediction and database commit."""
        return predictor.predict_single_flow(
            flow,
            source_ip=src_ip,
            destination_ip=dst_ip,
            log_to_db=True
        )

    async def _simulation_step(self, predictor: predict.NIDSPredictor) -> Optional[Dict[str, Any]]:
        """Executes a single flow generation in thread executor and notifies listeners."""
        flow, src_ip, dst_ip = self._get_next_flow()
        if not flow:
            return None

        flow_with_noise = self._apply_subtle_noise(flow)

        # Execute prediction through standard NIDS inference pipeline (~0.3ms latency)
        result = self._predict_and_store_sync(predictor, flow_with_noise, src_ip, dst_ip)

        # Update diagnostics
        self.last_tick_time = time.time()
        self.last_tick_at = datetime.now(timezone.utc).isoformat()
        self.flows_generated += 1
        self.last_error = None

        if self.flows_generated % 50 == 0:
            logger.info(f"[Traffic Simulator] Progress: {self.flows_generated} flows generated and evaluated.")

        # Broadcast live prediction to connected SSE clients
        try:
            await stream_broadcaster.broadcast("prediction", {
                "id": result.get("record_id"),
                "category": result.get("prediction"),
                "status": result.get("status"),
                "severity": result.get("severity"),
                "confidence": result.get("confidence"),
                "source_ip": src_ip,
                "destination_ip": dst_ip,
                "is_anomaly": result.get("is_anomaly")
            })

            # Broadcast alert if threat detected
            if result.get("status") == "ATTACK" or result.get("is_anomaly"):
                await stream_broadcaster.broadcast("alert", {
                    "attack_type": result.get("prediction"),
                    "alert_level": result.get("severity"),
                    "confidence": result.get("confidence"),
                    "source_ip": src_ip,
                    "destination_ip": dst_ip,
                    "status": "NEW",
                    "description": f"Live SOC detection: {result.get('prediction')} from {src_ip}"
                })
        except Exception as e:
            logger.debug(f"[Traffic Simulator] Broadcast notice: {e}")

        return result

    def catch_up_sync(self, max_catchup: int = 30) -> int:
        """
        Synchronous safety net called by /api/stats and /alerts.
        If > 3 seconds elapsed since last tick (e.g. Render was asleep or background task paused),
        generates missed flows up to max_catchup through the real model immediately.
        """
        if not SIMULATOR_ENABLED:
            return 0

        now = time.time()
        elapsed = now - self.last_tick_time
        if elapsed < 3.0:
            return 0

        rate = max(0.5, SIMULATOR_RATE)
        missed_count = min(max_catchup, max(1, int(elapsed / rate)))
        self.last_tick_time = now

        predictor = predict.get_predictor()
        count = 0
        for _ in range(missed_count):
            flow, src_ip, dst_ip = self._get_next_flow()
            if not flow:
                break
            flow_noised = self._apply_subtle_noise(flow)
            self._predict_and_store_sync(predictor, flow_noised, src_ip, dst_ip)
            self.flows_generated += 1
            count += 1

        if count > 0:
            self.last_tick_at = datetime.now(timezone.utc).isoformat()
            logger.info(f"[Traffic Simulator] Catch-up executed: generated {count} missed flows (idle: {elapsed:.1f}s).")

        return count

    async def run_loop(self) -> None:
        """Continuous background loop dispatching simulated traffic."""
        sample_name = self.active_sample_path.name if self.active_sample_path else "unknown"
        total_rows = len(self.benign_pool) + sum(len(v) for v in self.attack_pool.values())
        logger.info(
            f"Simulator started, rate={SIMULATOR_RATE}s, sample file={sample_name}, rows={total_rows}"
        )
        predictor = predict.get_predictor()

        while self.running:
            try:
                interval = max(0.5, SIMULATOR_RATE * random.uniform(0.7, 1.4))
                await asyncio.sleep(interval)

                if SIMULATOR_ENABLED:
                    await self._simulation_step(predictor)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {str(e)}"
                logger.exception(f"[Traffic Simulator] Error during simulation step: {e}")
                await asyncio.sleep(2.0)

        logger.info("[Traffic Simulator] Loop terminated cleanly.")

    def start(self) -> None:
        """Launches the simulator background task in the running asyncio loop with guard."""
        if self._task and not self._task.done():
            logger.info("[Traffic Simulator] Task already running, skipping dispatch.")
            return

        self.running = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        self._task = loop.create_task(self.run_loop())
        logger.info("[Traffic Simulator] Background task dispatched.")

    async def stop_async(self) -> None:
        """Cancels and awaits clean termination of the background task."""
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("[Traffic Simulator] Background task cancelled and joined.")

    def stop(self) -> None:
        """Cancels and halts the background task."""
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None
            logger.info("[Traffic Simulator] Background task cancelled.")


# Global simulator singleton
traffic_simulator = NetworkTrafficSimulator()
