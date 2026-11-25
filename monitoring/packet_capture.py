"""
Network Intrusion Detection System (NIDS) - Live & Offline Packet Capture
Captures live packets or reads offline PCAP files via Scapy and aggregates them into
CICFlowMeter-aligned bidirectional flows.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List, Tuple, Union
from monitoring.flow_extractor import NetworkFlow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _parse_ip_transport_sizes(pkt) -> Tuple[int, int]:
    """
    Computes (payload_bytes, transport_header_bytes) matching CICFlowMeter's
    BasicPacketInfo.java rules (ignoring L2 Ethernet headers and zero-padding).
    """
    ip = pkt["IP"]
    ip_hdr_len = int(ip.ihl * 4) if getattr(ip, "ihl", None) else 20
    ip_total_len = int(ip.len) if getattr(ip, "len", None) else len(bytes(ip))

    if pkt.haslayer("TCP"):
        tcp = pkt["TCP"]
        trans_hdr_len = int(tcp.dataofs * 4) if getattr(tcp, "dataofs", None) else 20
    elif pkt.haslayer("UDP"):
        trans_hdr_len = 8
    elif pkt.haslayer("ICMP"):
        trans_hdr_len = 8
    else:
        trans_hdr_len = 0

    payload_bytes = max(0, ip_total_len - ip_hdr_len - trans_hdr_len)
    return payload_bytes, trans_hdr_len


class FlowAggregator:
    """Tracks active network flows and flushes expired flows for SVM prediction."""

    def __init__(
        self,
        flow_timeout_seconds: float = 120.0,
        on_flow_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.flow_timeout = flow_timeout_seconds
        self.on_flow_complete = on_flow_complete
        self.active_flows: Dict[str, NetworkFlow] = {}

    def process_packet(self, pkt) -> None:
        """Parses a captured packet, determines flow key, and updates flow state."""
        try:
            if not pkt.haslayer("IP"):
                return

            ip = pkt["IP"]
            src_ip = ip.src
            dst_ip = ip.dst
            proto_name = "TCP" if pkt.haslayer("TCP") else ("UDP" if pkt.haslayer("UDP") else "OTHER")

            src_port = 0
            dst_port = 0
            window_size: Optional[int] = None
            flags = {"SYN": 0, "ACK": 0, "FIN": 0, "RST": 0, "PSH": 0, "URG": 0, "CWE": 0, "ECE": 0}

            if pkt.haslayer("TCP"):
                tcp = pkt["TCP"]
                src_port = int(tcp.sport)
                dst_port = int(tcp.dport)
                window_size = int(tcp.window)
                flag_str = str(tcp.flags)
                flags["SYN"] = 1 if "S" in flag_str else 0
                flags["ACK"] = 1 if "A" in flag_str else 0
                flags["FIN"] = 1 if "F" in flag_str else 0
                flags["RST"] = 1 if "R" in flag_str else 0
                flags["PSH"] = 1 if "P" in flag_str else 0
                flags["URG"] = 1 if "U" in flag_str else 0
                flags["CWE"] = 1 if "C" in flag_str else 0
                flags["ECE"] = 1 if "E" in flag_str else 0
            elif pkt.haslayer("UDP"):
                udp = pkt["UDP"]
                src_port = int(udp.sport)
                dst_port = int(udp.dport)

            # Use packet capture timestamp in microseconds (avoids float64 epoch subtraction cancellation)
            pkt_ts = getattr(pkt, "time", None)
            if pkt_ts is not None:
                now = float(pkt_ts)
                now_us = float(pkt_ts * 1_000_000)
            else:
                now = time.time()
                now_us = now * 1e6

            payload_bytes, header_bytes = _parse_ip_transport_sizes(pkt)
            total_len = len(pkt)

            # Flush any flows that timed out prior to this packet
            self.flush_expired_flows(now)

            fwd_key = f"{src_ip}:{src_port}->{dst_ip}:{dst_port}:{proto_name}"
            bwd_key = f"{dst_ip}:{dst_port}->{src_ip}:{src_port}:{proto_name}"

            if fwd_key in self.active_flows:
                flow = self.active_flows[fwd_key]
                flow.add_packet(
                    length=total_len,
                    timestamp=now,
                    is_forward=True,
                    flags=flags,
                    payload_length=payload_bytes,
                    header_length=header_bytes,
                    window_size=window_size,
                    timestamp_us=now_us,
                )
            elif bwd_key in self.active_flows:
                flow = self.active_flows[bwd_key]
                flow.add_packet(
                    length=total_len,
                    timestamp=now,
                    is_forward=False,
                    flags=flags,
                    payload_length=payload_bytes,
                    header_length=header_bytes,
                    window_size=window_size,
                    timestamp_us=now_us,
                )
            else:
                tuple_key = (src_ip, dst_ip, src_port, dst_port, proto_name)
                flow = NetworkFlow(tuple_key, now)
                flow.add_packet(
                    length=total_len,
                    timestamp=now,
                    is_forward=True,
                    flags=flags,
                    payload_length=payload_bytes,
                    header_length=header_bytes,
                    window_size=window_size,
                    timestamp_us=now_us,
                )
                self.active_flows[fwd_key] = flow

        except Exception as e:
            logger.debug(f"Packet parsing error: {e}")

    def flush_expired_flows(self, current_time: float) -> List[Dict[str, Any]]:
        """Flushes flows that have exceeded the inactivity timeout."""
        expired_keys = []
        flushed: List[Dict[str, Any]] = []
        for key, flow in self.active_flows.items():
            if (current_time - flow.last_seen) > self.flow_timeout:
                expired_keys.append(key)
                features = flow.extract_features()
                flushed.append(features)
                if self.on_flow_complete:
                    self.on_flow_complete(features)

        for k in expired_keys:
            del self.active_flows[k]
        return flushed

    def flush_all(self) -> List[Dict[str, Any]]:
        """Flushes all remaining active flows (e.g., at end of pcap)."""
        return self.flush_expired_flows(float("inf"))


def extract_flows_from_pcap(
    pcap_path: Union[str, Path],
    flow_timeout_seconds: float = 120.0,
) -> List[Dict[str, Any]]:
    """
    Reads packets from a PCAP file via Scapy using wire timestamps and extracts
    CICFlowMeter-aligned bidirectional flow features.
    """
    from scapy.all import rdpcap

    completed: List[Dict[str, Any]] = []
    aggregator = FlowAggregator(
        flow_timeout_seconds=flow_timeout_seconds,
        on_flow_complete=completed.append,
    )
    packets = rdpcap(str(pcap_path))
    for pkt in packets:
        aggregator.process_packet(pkt)
    aggregator.flush_all()
    return completed


def start_live_capture(interface: Optional[str] = None, packet_count: int = 50, callback: Optional[Callable] = None):
    """Starts capturing authorized network packets."""
    try:
        from scapy.all import sniff
        aggregator = FlowAggregator(flow_timeout_seconds=120.0, on_flow_complete=callback)
        logger.info(f"Starting Scapy packet capture on {interface or 'default interface'}...")
        sniff(iface=interface, prn=aggregator.process_packet, count=packet_count, store=False)
        aggregator.flush_all()
    except ImportError:
        logger.error("Scapy is not installed. Run: pip install scapy")
    except Exception as e:
        logger.error(f"Live sniffing failed: {e}. Note: On Windows, Npcap driver and Administrator rights may be required.")
