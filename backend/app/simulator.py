"""
Real-Time Background Network Traffic Simulator for NIDS SOC
Samples genuine CIC-IDS2017 benchmark flows, introduces subtle variance,
executes model inference via the standard prediction pipeline, and updates database records.
"""

import os
import random
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

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

    def __init__(self, sample_path: Path = config.DATASET_RAW_DIR / "CICIDS2017_Benchmark_Sample.csv"):
        self.sample_path = sample_path
        self.benign_pool: List[Dict[str, Any]] = []
        self.attack_pool: Dict[str, List[Dict[str, Any]]] = {}
        self.running: bool = False
        self._task: Optional[asyncio.Task] = None
        self._burst_counter: int = 0
        self._active_burst_category: Optional[str] = None
        self._active_burst_attacker_ip: Optional[str] = None
        self._burst_remaining: int = 0
        self._load_pools()

    def _load_pools(self) -> None:
        """Loads and indexes sample flows from the CIC-IDS2017 held-out sample."""
        if not self.sample_path.exists():
            logger.warning(f"[Traffic Simulator] Sample file {self.sample_path} not found. Simulator inactive.")
            return

        try:
            df = pd.read_csv(self.sample_path, nrows=3000)
            label_col = [c for c in df.columns if "label" in c.lower()]
            if not label_col:
                logger.error("[Traffic Simulator] Missing label column in benchmark file.")
                return

            label_col_name = label_col[0]

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
                f"[Traffic Simulator] Indexed {len(self.benign_pool)} benign flows and "
                f"{total_attacks} attacks across {len(self.attack_pool)} categories."
            )
        except Exception as e:
            logger.error(f"[Traffic Simulator] Failed to initialize sample pools: {e}")

    def _get_next_flow(self) -> tuple[Dict[str, Any], str, str]:
        """
        Picks the next flow vector, source IP, and target IP based on realistic distribution
        or ongoing attack burst.
        """
        # Handle active burst
        if self._burst_remaining > 0 and self._active_burst_category and self._active_burst_attacker_ip:
            self._burst_remaining -= 1
            pool = self.attack_pool.get(self._active_burst_category)
            if pool:
                flow = dict(random.choice(pool))
            else:
                flow = dict(random.choice(self.benign_pool))
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
                # Add +/- 1% to 2% random variation
                factor = 1.0 + random.uniform(-0.02, 0.02)
                noised[k] = max(0.0, float(v) * factor)
        return noised

    async def _simulation_step(self, predictor: predict.NIDSPredictor) -> None:
        """Executes a single flow generation, model inference, and notification."""
        flow, src_ip, dst_ip = self._get_next_flow()
        if not flow:
            return

        flow_with_noise = self._apply_subtle_noise(flow)

        # Execute prediction through standard NIDS inference pipeline
        result = predictor.predict_single_flow(
            flow_with_noise,
            source_ip=src_ip,
            destination_ip=dst_ip,
            log_to_db=True
        )

        # Broadcast live prediction to connected SSE / WebSocket clients
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

    async def run_loop(self) -> None:
        """Continuous background loop dispatching simulated traffic."""
        logger.info(f"[Traffic Simulator] Started. Rate: ~{SIMULATOR_RATE}s. Enabled: {SIMULATOR_ENABLED}")
        predictor = predict.get_predictor()

        while self.running:
            try:
                # Random jitter around SIMULATOR_RATE (between 1.0s and 3.0s)
                interval = max(0.5, SIMULATOR_RATE * random.uniform(0.7, 1.4))
                await asyncio.sleep(interval)

                if SIMULATOR_ENABLED:
                    await self._simulation_step(predictor)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Traffic Simulator] Error during simulation step: {e}")
                await asyncio.sleep(2.0)

        logger.info("[Traffic Simulator] Loop terminated cleanly.")

    def start(self) -> None:
        """Launches the simulator background task in the running asyncio loop."""
        if self._task is None or self._task.done():
            self.running = True
            loop = asyncio.get_event_loop()
            self._task = loop.create_task(self.run_loop())
            logger.info("[Traffic Simulator] Background task dispatched.")

    def stop(self) -> None:
        """Cancels and halts the background task."""
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info("[Traffic Simulator] Background task cancelled.")


# Global simulator singleton
traffic_simulator = NetworkTrafficSimulator()
