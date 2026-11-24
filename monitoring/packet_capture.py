"""
Network Intrusion Detection System (NIDS) - Live Packet Capture
Captures network packets using Scapy and aggregates them into flows.
"""

import time
import logging
from typing import Dict, Any, Callable, Optional
from monitoring.flow_extractor import NetworkFlow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FlowAggregator:
    """Tracks active network flows and flushes expired flows for SVM prediction."""
    def __init__(self, flow_timeout_seconds: float = 10.0, on_flow_complete: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.flow_timeout = flow_timeout_seconds
        self.on_flow_complete = on_flow_complete
        self.active_flows: Dict[str, NetworkFlow] = {}

    def process_packet(self, pkt) -> None:
        """Parses a captured packet, determines flow key, and updates flow state."""
        try:
            # Check for IP layer
            if not pkt.haslayer("IP"):
                return
                
            ip = pkt["IP"]
            src_ip = ip.src
            dst_ip = ip.dst
            proto_name = "TCP" if pkt.haslayer("TCP") else ("UDP" if pkt.haslayer("UDP") else "OTHER")
            
            src_port = 0
            dst_port = 0
            flags = {"SYN": 0, "ACK": 0, "FIN": 0, "RST": 0, "PSH": 0, "URG": 0}
            
            if pkt.haslayer("TCP"):
                tcp = pkt["TCP"]
                src_port = tcp.sport
                dst_port = tcp.dport
                # Parse TCP flags
                flag_str = str(tcp.flags)
                flags["SYN"] = 1 if "S" in flag_str else 0
                flags["ACK"] = 1 if "A" in flag_str else 0
                flags["FIN"] = 1 if "F" in flag_str else 0
                flags["RST"] = 1 if "R" in flag_str else 0
                flags["PSH"] = 1 if "P" in flag_str else 0
                flags["URG"] = 1 if "U" in flag_str else 0
            elif pkt.haslayer("UDP"):
                udp = pkt["UDP"]
                src_port = udp.sport
                dst_port = udp.dport
                
            now = time.time()
            length = len(pkt)
            
            # Canonical forward/backward key ordering
            fwd_key = f"{src_ip}:{src_port}->{dst_ip}:{dst_port}:{proto_name}"
            bwd_key = f"{dst_ip}:{dst_port}->{src_ip}:{src_port}:{proto_name}"
            
            if fwd_key in self.active_flows:
                flow = self.active_flows[fwd_key]
                flow.add_packet(length, now, is_forward=True, flags=flags)
            elif bwd_key in self.active_flows:
                flow = self.active_flows[bwd_key]
                flow.add_packet(length, now, is_forward=False, flags=flags)
            else:
                # New flow initiated
                tuple_key = (src_ip, dst_ip, src_port, dst_port, proto_name)
                flow = NetworkFlow(tuple_key, now)
                flow.add_packet(length, now, is_forward=True, flags=flags)
                self.active_flows[fwd_key] = flow
                
            # Periodically flush expired flows
            self.flush_expired_flows(now)
            
        except Exception as e:
            logger.debug(f"Packet parsing error: {e}")

    def flush_expired_flows(self, current_time: float) -> None:
        """Flushes flows that have exceeded the inactivity timeout."""
        expired_keys = []
        for key, flow in self.active_flows.items():
            if (current_time - flow.last_seen) > self.flow_timeout:
                expired_keys.append(key)
                if self.on_flow_complete:
                    features = flow.extract_features()
                    self.on_flow_complete(features)
                    
        for k in expired_keys:
            del self.active_flows[k]


def start_live_capture(interface: Optional[str] = None, packet_count: int = 50, callback: Optional[Callable] = None):
    """Starts capturing authorized network packets."""
    try:
        from scapy.all import sniff
        aggregator = FlowAggregator(flow_timeout_seconds=5.0, on_flow_complete=callback)
        logger.info(f"Starting Scapy packet capture on {interface or 'default interface'}...")
        sniff(iface=interface, prn=aggregator.process_packet, count=packet_count, store=False)
        # Flush remaining flows
        aggregator.flush_expired_flows(time.time() + 100)
    except ImportError:
        logger.error("Scapy is not installed. Run: pip install scapy")
    except Exception as e:
        logger.error(f"Live sniffing failed: {e}. Note: On Windows, Npcap driver and Administrator rights may be required.")
