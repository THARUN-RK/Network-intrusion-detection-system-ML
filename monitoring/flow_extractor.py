"""
Network Intrusion Detection System (NIDS) - Flow Feature Extractor
Converts raw captured packets into bidirectional flow records matching the 71-feature CIC-IDS2017 schema.
"""

import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


class NetworkFlow:
    """Represents a bidirectional network flow between two endpoints."""
    def __init__(self, key: Tuple[str, str, int, int, str], start_time: float):
        self.key = key  # (src_ip, dst_ip, src_port, dst_port, protocol)
        self.start_time = start_time
        self.last_seen = start_time
        
        # Forward and Backward metrics
        self.fwd_packets = 0
        self.bwd_packets = 0
        self.fwd_bytes = 0
        self.bwd_bytes = 0
        
        self.fwd_lengths = []
        self.bwd_lengths = []
        
        # Inter-arrival times
        self.last_fwd_time = start_time
        self.last_bwd_time = start_time
        self.fwd_iats = []
        self.bwd_iats = []
        self.flow_iats = []
        
        # TCP Flags
        self.syn_count = 0
        self.ack_count = 0
        self.fin_count = 0
        self.rst_count = 0
        self.psh_count = 0
        self.urg_count = 0

    def add_packet(self, length: int, timestamp: float, is_forward: bool, flags: Dict[str, int]) -> None:
        duration_from_last = max(0.0, (timestamp - self.last_seen) * 1e6)  # microseconds
        self.flow_iats.append(duration_from_last)
        self.last_seen = timestamp
        
        if is_forward:
            self.fwd_packets += 1
            self.fwd_bytes += length
            self.fwd_lengths.append(length)
            if self.fwd_packets > 1:
                self.fwd_iats.append(max(0.0, (timestamp - self.last_fwd_time) * 1e6))
            self.last_fwd_time = timestamp
        else:
            self.bwd_packets += 1
            self.bwd_bytes += length
            self.bwd_lengths.append(length)
            if self.bwd_packets > 1:
                self.bwd_iats.append(max(0.0, (timestamp - self.last_bwd_time) * 1e6))
            self.last_bwd_time = timestamp
            
        self.syn_count += flags.get("SYN", 0)
        self.ack_count += flags.get("ACK", 0)
        self.fin_count += flags.get("FIN", 0)
        self.rst_count += flags.get("RST", 0)
        self.psh_count += flags.get("PSH", 0)
        self.urg_count += flags.get("URG", 0)

    def extract_features(self) -> Dict[str, Any]:
        """Converts flow state into the canonical CIC-IDS2017 schema."""
        duration_us = max(1.0, (self.last_seen - self.start_time) * 1e6)
        total_packets = self.fwd_packets + self.bwd_packets
        total_bytes = self.fwd_bytes + self.bwd_bytes
        
        all_lengths = self.fwd_lengths + self.bwd_lengths
        pkt_len_mean = np.mean(all_lengths) if all_lengths else 0.0
        pkt_len_std = np.std(all_lengths) if all_lengths else 0.0
        
        fwd_mean = np.mean(self.fwd_lengths) if self.fwd_lengths else 0.0
        bwd_mean = np.mean(self.bwd_lengths) if self.bwd_lengths else 0.0
        
        flow_bytes_s = (total_bytes / (duration_us / 1e6)) if duration_us > 0 else 0.0
        flow_pkts_s = (total_packets / (duration_us / 1e6)) if duration_us > 0 else 0.0
        
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
            "Fwd Packet Length Max": max(self.fwd_lengths, default=0),
            "Fwd Packet Length Min": min(self.fwd_lengths, default=0),
            "Fwd Packet Length Mean": fwd_mean,
            "Fwd Packet Length Std": np.std(self.fwd_lengths) if self.fwd_lengths else 0.0,
            "Bwd Packet Length Max": max(self.bwd_lengths, default=0),
            "Bwd Packet Length Min": min(self.bwd_lengths, default=0),
            "Bwd Packet Length Mean": bwd_mean,
            "Bwd Packet Length Std": np.std(self.bwd_lengths) if self.bwd_lengths else 0.0,
            "Flow Bytes/s": flow_bytes_s,
            "Flow Packets/s": flow_pkts_s,
            "Flow IAT Mean": np.mean(self.flow_iats) if self.flow_iats else 0.0,
            "Flow IAT Std": np.std(self.flow_iats) if self.flow_iats else 0.0,
            "Flow IAT Max": max(self.flow_iats, default=0),
            "Flow IAT Min": min(self.flow_iats, default=0),
            "Fwd IAT Total": sum(self.fwd_iats),
            "Fwd IAT Mean": np.mean(self.fwd_iats) if self.fwd_iats else 0.0,
            "Fwd IAT Std": np.std(self.fwd_iats) if self.fwd_iats else 0.0,
            "Fwd IAT Max": max(self.fwd_iats, default=0),
            "Fwd IAT Min": min(self.fwd_iats, default=0),
            "Bwd IAT Total": sum(self.bwd_iats),
            "Bwd IAT Mean": np.mean(self.bwd_iats) if self.bwd_iats else 0.0,
            "Bwd IAT Std": np.std(self.bwd_iats) if self.bwd_iats else 0.0,
            "Bwd IAT Max": max(self.bwd_iats, default=0),
            "Bwd IAT Min": min(self.bwd_iats, default=0),
            "Fwd PSH Flags": self.psh_count,
            "Bwd PSH Flags": 0,
            "Fwd URG Flags": self.urg_count,
            "Bwd URG Flags": 0,
            "Fwd Header Length": self.fwd_packets * 20,
            "Bwd Header Length": self.bwd_packets * 20,
            "Fwd Packets/s": (self.fwd_packets / (duration_us / 1e6)) if duration_us > 0 else 0.0,
            "Bwd Packets/s": (self.bwd_packets / (duration_us / 1e6)) if duration_us > 0 else 0.0,
            "Min Packet Length": min(all_lengths, default=0),
            "Max Packet Length": max(all_lengths, default=0),
            "Packet Length Mean": pkt_len_mean,
            "Packet Length Std": pkt_len_std,
            "Packet Length Variance": pkt_len_std ** 2,
            "FIN Flag Count": self.fin_count,
            "SYN Flag Count": self.syn_count,
            "RST Flag Count": self.rst_count,
            "PSH Flag Count": self.psh_count,
            "ACK Flag Count": self.ack_count,
            "URG Flag Count": self.urg_count,
            "CWE Flag Count": 0,
            "ECE Flag Count": 0,
            "Down/Up Ratio": (self.bwd_packets / self.fwd_packets) if self.fwd_packets > 0 else 0.0,
            "Average Packet Size": pkt_len_mean,
            "Avg Fwd Segment Size": fwd_mean,
            "Avg Bwd Segment Size": bwd_mean,
            "Subflow Fwd Packets": self.fwd_packets,
            "Subflow Fwd Bytes": self.fwd_bytes,
            "Subflow Bwd Packets": self.bwd_packets,
            "Subflow Bwd Bytes": self.bwd_bytes,
            "Init_Win_bytes_forward": 65535,
            "Init_Win_bytes_backward": 65535,
            "act_data_pkt_fwd": max(0, self.fwd_packets - 1),
            "min_seg_size_forward": 20,
            "Active Mean": 0.0,
            "Active Std": 0.0,
            "Active Max": 0.0,
            "Active Min": 0.0,
            "Idle Mean": 0.0,
            "Idle Std": 0.0,
            "Idle Max": 0.0,
            "Idle Min": 0.0
        }
