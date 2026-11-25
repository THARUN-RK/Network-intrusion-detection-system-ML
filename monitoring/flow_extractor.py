"""
Network Intrusion Detection System (NIDS) - CICFlowMeter-Aligned Flow Feature Extractor
Converts raw captured packets into bidirectional flow records matching the 71-feature CIC-IDS2017 schema.

Alignment notes vs CICFlowMeter (ISCXFlowMeter / BasicFlow.java):
- Uses transport payload bytes (excluding L2 Ethernet + L3 IP + L4 TCP/UDP headers) when provided
  by packet parsers (falls back to `length` only if callers supply pre-computed payload sizes directly).
- Uses sample standard deviation (ddof=1) matching Apache Commons Math `SummaryStatistics`.
- Only records Flow IAT starting from the 2nd packet in the flow (K-1 intervals for K packets).
- Separates forward vs backward PSH and URG flags, and counts CWE and ECE TCP flags.
- Computes actual transport header lengths (`Fwd/Bwd Header Length`, `min_seg_size_forward`),
  `act_data_pkt_fwd` (forward packets with payload > 0), initial TCP windows (`Init_Win_bytes_*`),
  subflow counts (1s subflow threshold), and Active/Idle periods (5s activity timeout).
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np


ACTIVITY_TIMEOUT_US = 5_000_000.0   # 5.0 seconds in microseconds (CICFlowMeter default)
SUBFLOW_TIMEOUT_US = 1_000_000.0    # 1.0 second in microseconds (CICFlowMeter default)


def _sample_std(values: List[float]) -> float:
    """Bessel-corrected sample standard deviation matching CICFlowMeter's SummaryStatistics."""
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


class NetworkFlow:
    """Represents a bidirectional network flow between two endpoints."""

    def __init__(
        self,
        key: Tuple[str, str, int, int, str],
        start_time: float,
        activity_timeout_us: float = ACTIVITY_TIMEOUT_US,
        subflow_timeout_us: float = SUBFLOW_TIMEOUT_US,
    ):
        self.key = key  # (src_ip, dst_ip, src_port, dst_port, protocol)
        self.start_time_us = float(start_time) * 1e6
        self.last_seen_us = self.start_time_us
        self.start_time = float(start_time)
        self.last_seen = float(start_time)
        self.activity_timeout_us = float(activity_timeout_us)
        self.subflow_timeout_us = float(subflow_timeout_us)

        # Forward and Backward packet/payload counters
        self.fwd_packets = 0
        self.bwd_packets = 0
        self.fwd_bytes = 0
        self.bwd_bytes = 0
        self.fwd_data_packets = 0  # act_data_pkt_fwd: forward packets with >= 1 payload byte

        self.fwd_lengths: List[float] = []
        self.bwd_lengths: List[float] = []
        self.fwd_header_lengths: List[int] = []
        self.bwd_header_lengths: List[int] = []

        # Initial TCP window sizes (-1 when uninitialized / non-TCP, per CICFlowMeter)
        self.init_win_bytes_fwd: Optional[int] = None
        self.init_win_bytes_bwd: Optional[int] = None

        # Inter-arrival times (microseconds)
        self.last_fwd_us = self.start_time_us
        self.last_bwd_us = self.start_time_us
        self.fwd_iats: List[float] = []
        self.bwd_iats: List[float] = []
        self.flow_iats: List[float] = []

        # Subflow & Active/Idle state (microseconds)
        self.start_active_us = self.start_time_us
        self.end_active_us = self.start_time_us
        self.active_durations_us: List[float] = []
        self.idle_durations_us: List[float] = []
        self.sf_last_packet_us: Optional[float] = None
        self.sf_count = 0

        # TCP Flags (total + directional PSH/URG)
        self.syn_count = 0
        self.ack_count = 0
        self.fin_count = 0
        self.rst_count = 0
        self.psh_count = 0
        self.urg_count = 0
        self.cwe_count = 0
        self.ece_count = 0
        self.fwd_psh_flags = 0
        self.bwd_psh_flags = 0
        self.fwd_urg_flags = 0
        self.bwd_urg_flags = 0

    def add_packet(
        self,
        length: int,
        timestamp: float,
        is_forward: bool,
        flags: Dict[str, int],
        payload_length: Optional[int] = None,
        header_length: Optional[int] = None,
        window_size: Optional[int] = None,
        timestamp_us: Optional[float] = None,
    ) -> None:
        ts = float(timestamp)
        ts_us = float(timestamp_us) if timestamp_us is not None else (ts * 1e6)
        pkt_payload_len = int(length if payload_length is None else max(0, payload_length))
        pkt_hdr_len = int(20 if header_length is None else max(0, header_length))

        total_seen_before = self.fwd_packets + self.bwd_packets
        if total_seen_before == 0:
            # First packet anchors flow timing cleanly in microseconds (prevents float64 epoch cancellation)
            self.start_time = ts
            self.last_seen = ts
            self.start_time_us = ts_us
            self.last_seen_us = ts_us
            self.start_active_us = ts_us
            self.end_active_us = ts_us
        else:
            duration_from_last = max(0.0, ts_us - self.last_seen_us)
            self.flow_iats.append(duration_from_last)
            self.last_seen = max(self.last_seen, ts)
            self.last_seen_us = max(self.last_seen_us, ts_us)

            # CICFlowMeter Active/Idle update
            if (ts_us - self.end_active_us) > self.activity_timeout_us:
                active_span = self.end_active_us - self.start_active_us
                if active_span > 0:
                    self.active_durations_us.append(active_span)
                self.idle_durations_us.append(ts_us - self.end_active_us)
                self.start_active_us = ts_us
                self.end_active_us = ts_us
            else:
                self.end_active_us = ts_us

        # Subflow counter (CICFlowMeter 1.0s subflow gap threshold)
        if self.sf_last_packet_us is None:
            self.sf_count = 1
        elif (ts_us - self.sf_last_packet_us) > self.subflow_timeout_us:
            self.sf_count += 1
        self.sf_last_packet_us = ts_us

        psh_val = int(flags.get("PSH", 0))
        urg_val = int(flags.get("URG", 0))

        if is_forward:
            self.fwd_packets += 1
            self.fwd_bytes += pkt_payload_len
            self.fwd_lengths.append(float(pkt_payload_len))
            self.fwd_header_lengths.append(pkt_hdr_len)
            if pkt_payload_len > 0:
                self.fwd_data_packets += 1
            if window_size is not None and self.init_win_bytes_fwd is None:
                self.init_win_bytes_fwd = int(window_size)
            if self.fwd_packets > 1:
                self.fwd_iats.append(max(0.0, ts_us - self.last_fwd_us))
            self.last_fwd_us = ts_us
            self.fwd_psh_flags += psh_val
            self.fwd_urg_flags += urg_val
        else:
            self.bwd_packets += 1
            self.bwd_bytes += pkt_payload_len
            self.bwd_lengths.append(float(pkt_payload_len))
            self.bwd_header_lengths.append(pkt_hdr_len)
            if window_size is not None and self.init_win_bytes_bwd is None:
                self.init_win_bytes_bwd = int(window_size)
            if self.bwd_packets > 1:
                self.bwd_iats.append(max(0.0, ts_us - self.last_bwd_us))
            self.last_bwd_us = ts_us
            self.bwd_psh_flags += psh_val
            self.bwd_urg_flags += urg_val

        self.syn_count += int(flags.get("SYN", 0))
        self.ack_count += int(flags.get("ACK", 0))
        self.fin_count += int(flags.get("FIN", 0))
        self.rst_count += int(flags.get("RST", 0))
        self.psh_count += psh_val
        self.urg_count += urg_val
        self.cwe_count += int(flags.get("CWE", 0))
        self.ece_count += int(flags.get("ECE", 0))

    def extract_features(self) -> Dict[str, Any]:
        """Converts flow state into the canonical 71-feature CIC-IDS2017 schema."""
        duration_us = max(0.0, self.last_seen_us - self.start_time_us)
        duration_s = duration_us / 1e6
        total_packets = self.fwd_packets + self.bwd_packets
        total_bytes = self.fwd_bytes + self.bwd_bytes

        all_lengths = self.fwd_lengths + self.bwd_lengths
        pkt_len_mean = float(np.mean(all_lengths)) if all_lengths else 0.0
        pkt_len_std = _sample_std(all_lengths)

        fwd_mean = float(np.mean(self.fwd_lengths)) if self.fwd_lengths else 0.0
        bwd_mean = float(np.mean(self.bwd_lengths)) if self.bwd_lengths else 0.0

        flow_bytes_s = (total_bytes / duration_s) if duration_s > 0 else 0.0
        flow_pkts_s = (total_packets / duration_s) if duration_s > 0 else 0.0
        fwd_pkts_s = (self.fwd_packets / duration_s) if duration_s > 0 else 0.0
        bwd_pkts_s = (self.bwd_packets / duration_s) if duration_s > 0 else 0.0

        sf_div = max(1, self.sf_count)

        return {
            "source_ip": self.key[0],
            "destination_ip": self.key[1],
            "source_port": self.key[2],
            "protocol": self.key[4],
            "Destination Port": self.key[3],
            "Flow Duration": duration_us,
            "Total Fwd Packets": self.fwd_packets,
            "Total Backward Packets": self.bwd_packets,
            "Total Length of Fwd Packets": self.fwd_bytes,
            "Total Length of Bwd Packets": self.bwd_bytes,
            "Fwd Packet Length Max": max(self.fwd_lengths, default=0.0),
            "Fwd Packet Length Min": min(self.fwd_lengths, default=0.0),
            "Fwd Packet Length Mean": fwd_mean,
            "Fwd Packet Length Std": _sample_std(self.fwd_lengths),
            "Bwd Packet Length Max": max(self.bwd_lengths, default=0.0),
            "Bwd Packet Length Min": min(self.bwd_lengths, default=0.0),
            "Bwd Packet Length Mean": bwd_mean,
            "Bwd Packet Length Std": _sample_std(self.bwd_lengths),
            "Flow Bytes/s": flow_bytes_s,
            "Flow Packets/s": flow_pkts_s,
            "Flow IAT Mean": float(np.mean(self.flow_iats)) if self.flow_iats else 0.0,
            "Flow IAT Std": _sample_std(self.flow_iats),
            "Flow IAT Max": max(self.flow_iats, default=0.0),
            "Flow IAT Min": min(self.flow_iats, default=0.0),
            "Fwd IAT Total": float(sum(self.fwd_iats)),
            "Fwd IAT Mean": float(np.mean(self.fwd_iats)) if self.fwd_iats else 0.0,
            "Fwd IAT Std": _sample_std(self.fwd_iats),
            "Fwd IAT Max": max(self.fwd_iats, default=0.0),
            "Fwd IAT Min": min(self.fwd_iats, default=0.0),
            "Bwd IAT Total": float(sum(self.bwd_iats)),
            "Bwd IAT Mean": float(np.mean(self.bwd_iats)) if self.bwd_iats else 0.0,
            "Bwd IAT Std": _sample_std(self.bwd_iats),
            "Bwd IAT Max": max(self.bwd_iats, default=0.0),
            "Bwd IAT Min": min(self.bwd_iats, default=0.0),
            "Fwd PSH Flags": self.fwd_psh_flags,
            "Bwd PSH Flags": self.bwd_psh_flags,
            "Fwd URG Flags": self.fwd_urg_flags,
            "Bwd URG Flags": self.bwd_urg_flags,
            "Fwd Header Length": int(sum(self.fwd_header_lengths)),
            "Bwd Header Length": int(sum(self.bwd_header_lengths)),
            "Fwd Packets/s": fwd_pkts_s,
            "Bwd Packets/s": bwd_pkts_s,
            "Min Packet Length": min(all_lengths, default=0.0),
            "Max Packet Length": max(all_lengths, default=0.0),
            "Packet Length Mean": pkt_len_mean,
            "Packet Length Std": pkt_len_std,
            "Packet Length Variance": pkt_len_std ** 2,
            "FIN Flag Count": self.fin_count,
            "SYN Flag Count": self.syn_count,
            "RST Flag Count": self.rst_count,
            "PSH Flag Count": self.psh_count,
            "ACK Flag Count": self.ack_count,
            "URG Flag Count": self.urg_count,
            "CWE Flag Count": self.cwe_count,
            "ECE Flag Count": self.ece_count,
            "Down/Up Ratio": (self.bwd_packets / self.fwd_packets) if self.fwd_packets > 0 else 0.0,
            "Average Packet Size": (total_bytes / total_packets) if total_packets > 0 else 0.0,
            "Avg Fwd Segment Size": fwd_mean,
            "Avg Bwd Segment Size": bwd_mean,
            "Subflow Fwd Packets": self.fwd_packets / sf_div,
            "Subflow Fwd Bytes": self.fwd_bytes / sf_div,
            "Subflow Bwd Packets": self.bwd_packets / sf_div,
            "Subflow Bwd Bytes": self.bwd_bytes / sf_div,
            "Init_Win_bytes_forward": -1 if self.init_win_bytes_fwd is None else self.init_win_bytes_fwd,
            "Init_Win_bytes_backward": -1 if self.init_win_bytes_bwd is None else self.init_win_bytes_bwd,
            "act_data_pkt_fwd": self.fwd_data_packets,
            "min_seg_size_forward": min(self.fwd_header_lengths, default=0),
            "Active Mean": float(np.mean(self.active_durations_us)) if self.active_durations_us else 0.0,
            "Active Std": _sample_std(self.active_durations_us),
            "Active Max": max(self.active_durations_us, default=0.0),
            "Active Min": min(self.active_durations_us, default=0.0),
            "Idle Mean": float(np.mean(self.idle_durations_us)) if self.idle_durations_us else 0.0,
            "Idle Std": _sample_std(self.idle_durations_us),
            "Idle Max": max(self.idle_durations_us, default=0.0),
            "Idle Min": min(self.idle_durations_us, default=0.0),
        }
