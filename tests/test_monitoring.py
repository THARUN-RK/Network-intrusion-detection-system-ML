"""
Unit tests for live monitoring, CICFlowMeter feature alignment, and PCAP extraction parity.
"""

from typing import Dict, Any, List
import numpy as np
import pytest

import config
from monitoring.flow_extractor import NetworkFlow, ACTIVITY_TIMEOUT_US, SUBFLOW_TIMEOUT_US
from monitoring.packet_capture import extract_flows_from_pcap
import predict


def test_network_flow_feature_extraction():
    key = ("192.168.1.10", "10.0.0.1", 54321, 80, "TCP")
    flow = NetworkFlow(key, start_time=100.0)

    # Forward packet (SYN)
    flow.add_packet(length=54, timestamp=100.0, is_forward=True, flags={"SYN": 1, "ACK": 0})
    # Backward packet (SYN-ACK)
    flow.add_packet(length=60, timestamp=100.02, is_forward=False, flags={"SYN": 1, "ACK": 1})
    # Forward packet (ACK + Data)
    flow.add_packet(length=500, timestamp=100.05, is_forward=True, flags={"SYN": 0, "ACK": 1, "PSH": 1})

    features = flow.extract_features()

    assert features["Destination Port"] == 80
    assert features["Total Fwd Packets"] == 2
    assert features["Total Backward Packets"] == 1
    assert features["Total Length of Fwd Packets"] == 554
    assert features["Total Length of Bwd Packets"] == 60
    assert features["SYN Flag Count"] == 2
    assert features["ACK Flag Count"] == 2
    assert features["PSH Flag Count"] == 1
    assert features["Flow Duration"] > 0
    assert len(features) >= 71
    assert features["source_ip"] == "192.168.1.10"
    assert features["destination_ip"] == "10.0.0.1"


def test_flow_features_compatibility_with_model():
    key = ("10.0.0.5", "192.168.1.1", 45000, 443, "TCP")
    flow = NetworkFlow(key, start_time=200.0)
    flow.add_packet(100, 200.1, True, {"SYN": 1})
    flow.add_packet(200, 200.2, False, {"ACK": 1})

    features = flow.extract_features()
    predictor = predict.get_predictor()
    result = predictor.predict_single_flow(features, log_to_db=False)

    assert "prediction" in result
    assert result["status"] in ["BENIGN", "ATTACK"]
    assert 0.0 <= result["confidence"] <= 100.0


def _cicflowmeter_reference_from_packets(packets: List[Any]) -> Dict[tuple, Dict[str, float]]:
    """
    Independent reference implementation of CICFlowMeter (ISCXFlowMeter BasicFlow.java)
    feature definitions computed directly from decoded PCAP packets.
    """
    flows: Dict[tuple, Dict[str, Any]] = {}

    def _sstd(vals: List[float]) -> float:
        return float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0

    for pkt in packets:
        ip = pkt["IP"]
        proto = "TCP" if pkt.haslayer("TCP") else ("UDP" if pkt.haslayer("UDP") else "OTHER")
        ip_total = int(ip.len)
        ip_hdr = int(ip.ihl * 4)

        if proto == "TCP":
            tcp = pkt["TCP"]
            sport, dport = int(tcp.sport), int(tcp.dport)
            trans_hdr = int(tcp.dataofs * 4)
            win = int(tcp.window)
            fl = str(tcp.flags)
        elif proto == "UDP":
            udp = pkt["UDP"]
            sport, dport = int(udp.sport), int(udp.dport)
            trans_hdr = 8
            win = None
            fl = ""
        else:
            sport, dport, trans_hdr, win, fl = 0, 0, 0, None, ""

        payload = max(0, ip_total - ip_hdr - trans_hdr)
        ts_us = float(pkt.time * 1_000_000)

        fwd_id = (ip.src, ip.dst, sport, dport, proto)
        bwd_id = (ip.dst, ip.src, dport, sport, proto)
        if fwd_id in flows:
            st = flows[fwd_id]
            is_fwd = True
        elif bwd_id in flows:
            st = flows[bwd_id]
            is_fwd = False
        else:
            st = {
                "dst_port": dport,
                "times": [], "fwd_times": [], "bwd_times": [],
                "fwd_lens": [], "bwd_lens": [],
                "fwd_hdrs": [], "bwd_hdrs": [],
                "win_fwd": None, "win_bwd": None,
                "fwd_data": 0,
                "fwd_psh": 0, "bwd_psh": 0, "fwd_urg": 0, "bwd_urg": 0,
                "flags": {k: 0 for k in ["F", "S", "R", "P", "A", "U", "C", "E"]},
                "active": [], "idle": [],
                "start_active": ts_us, "end_active": ts_us,
                "sf_last": None, "sf_cnt": 0,
            }
            flows[fwd_id] = st
            is_fwd = True

        if st["times"]:
            if (ts_us - st["end_active"]) > ACTIVITY_TIMEOUT_US:
                if (st["end_active"] - st["start_active"]) > 0:
                    st["active"].append(st["end_active"] - st["start_active"])
                st["idle"].append(ts_us - st["end_active"])
                st["start_active"] = ts_us
                st["end_active"] = ts_us
            else:
                st["end_active"] = ts_us

        st["sf_cnt"] = 1 if st["sf_last"] is None else st["sf_cnt"] + (1 if (ts_us - st["sf_last"]) > SUBFLOW_TIMEOUT_US else 0)
        st["sf_last"] = ts_us
        st["times"].append(ts_us)

        for k in st["flags"]:
            if k in fl:
                st["flags"][k] += 1

        if is_fwd:
            st["fwd_times"].append(ts_us)
            st["fwd_lens"].append(payload)
            st["fwd_hdrs"].append(trans_hdr)
            if payload > 0:
                st["fwd_data"] += 1
            if win is not None and st["win_fwd"] is None:
                st["win_fwd"] = win
            if "P" in fl:
                st["fwd_psh"] += 1
            if "U" in fl:
                st["fwd_urg"] += 1
        else:
            st["bwd_times"].append(ts_us)
            st["bwd_lens"].append(payload)
            st["bwd_hdrs"].append(trans_hdr)
            if win is not None and st["win_bwd"] is None:
                st["win_bwd"] = win
            if "P" in fl:
                st["bwd_psh"] += 1
            if "U" in fl:
                st["bwd_urg"] += 1

    out: Dict[tuple, Dict[str, float]] = {}
    for k, st in flows.items():
        dur_us = max(0.0, max(st["times"]) - min(st["times"]))
        dur_s = dur_us / 1e6
        all_lens = st["fwd_lens"] + st["bwd_lens"]
        f_iat = [st["times"][i] - st["times"][i - 1] for i in range(1, len(st["times"]))]
        fwd_iat = [st["fwd_times"][i] - st["fwd_times"][i - 1] for i in range(1, len(st["fwd_times"]))]
        bwd_iat = [st["bwd_times"][i] - st["bwd_times"][i - 1] for i in range(1, len(st["bwd_times"]))]
        nf = len(st["fwd_lens"])
        nb = len(st["bwd_lens"])
        sf = max(1, st["sf_cnt"])
        pkt_std = _sstd(all_lens)

        out[k] = {
            "Destination Port": st["dst_port"],
            "Flow Duration": dur_us,
            "Total Fwd Packets": nf,
            "Total Backward Packets": nb,
            "Total Length of Fwd Packets": sum(st["fwd_lens"]),
            "Total Length of Bwd Packets": sum(st["bwd_lens"]),
            "Fwd Packet Length Max": max(st["fwd_lens"], default=0),
            "Fwd Packet Length Min": min(st["fwd_lens"], default=0),
            "Fwd Packet Length Mean": float(np.mean(st["fwd_lens"])) if nf else 0.0,
            "Fwd Packet Length Std": _sstd(st["fwd_lens"]),
            "Bwd Packet Length Max": max(st["bwd_lens"], default=0),
            "Bwd Packet Length Min": min(st["bwd_lens"], default=0),
            "Bwd Packet Length Mean": float(np.mean(st["bwd_lens"])) if nb else 0.0,
            "Bwd Packet Length Std": _sstd(st["bwd_lens"]),
            "Flow Bytes/s": (sum(all_lens) / dur_s) if dur_s > 0 else 0.0,
            "Flow Packets/s": ((nf + nb) / dur_s) if dur_s > 0 else 0.0,
            "Flow IAT Mean": float(np.mean(f_iat)) if f_iat else 0.0,
            "Flow IAT Std": _sstd(f_iat),
            "Flow IAT Max": max(f_iat, default=0.0),
            "Flow IAT Min": min(f_iat, default=0.0),
            "Fwd IAT Total": sum(fwd_iat),
            "Fwd IAT Mean": float(np.mean(fwd_iat)) if fwd_iat else 0.0,
            "Fwd IAT Std": _sstd(fwd_iat),
            "Fwd IAT Max": max(fwd_iat, default=0.0),
            "Fwd IAT Min": min(fwd_iat, default=0.0),
            "Bwd IAT Total": sum(bwd_iat),
            "Bwd IAT Mean": float(np.mean(bwd_iat)) if bwd_iat else 0.0,
            "Bwd IAT Std": _sstd(bwd_iat),
            "Bwd IAT Max": max(bwd_iat, default=0.0),
            "Bwd IAT Min": min(bwd_iat, default=0.0),
            "Fwd PSH Flags": st["fwd_psh"],
            "Bwd PSH Flags": st["bwd_psh"],
            "Fwd URG Flags": st["fwd_urg"],
            "Bwd URG Flags": st["bwd_urg"],
            "Fwd Header Length": sum(st["fwd_hdrs"]),
            "Bwd Header Length": sum(st["bwd_hdrs"]),
            "Fwd Packets/s": (nf / dur_s) if dur_s > 0 else 0.0,
            "Bwd Packets/s": (nb / dur_s) if dur_s > 0 else 0.0,
            "Min Packet Length": min(all_lens, default=0),
            "Max Packet Length": max(all_lens, default=0),
            "Packet Length Mean": float(np.mean(all_lens)) if all_lens else 0.0,
            "Packet Length Std": pkt_std,
            "Packet Length Variance": pkt_std ** 2,
            "FIN Flag Count": st["flags"]["F"],
            "SYN Flag Count": st["flags"]["S"],
            "RST Flag Count": st["flags"]["R"],
            "PSH Flag Count": st["flags"]["P"],
            "ACK Flag Count": st["flags"]["A"],
            "URG Flag Count": st["flags"]["U"],
            "CWE Flag Count": st["flags"]["C"],
            "ECE Flag Count": st["flags"]["E"],
            "Down/Up Ratio": (nb / nf) if nf > 0 else 0.0,
            "Average Packet Size": (sum(all_lens) / (nf + nb)) if (nf + nb) else 0.0,
            "Avg Fwd Segment Size": float(np.mean(st["fwd_lens"])) if nf else 0.0,
            "Avg Bwd Segment Size": float(np.mean(st["bwd_lens"])) if nb else 0.0,
            "Subflow Fwd Packets": nf / sf,
            "Subflow Fwd Bytes": sum(st["fwd_lens"]) / sf,
            "Subflow Bwd Packets": nb / sf,
            "Subflow Bwd Bytes": sum(st["bwd_lens"]) / sf,
            "Init_Win_bytes_forward": -1 if st["win_fwd"] is None else st["win_fwd"],
            "Init_Win_bytes_backward": -1 if st["win_bwd"] is None else st["win_bwd"],
            "act_data_pkt_fwd": st["fwd_data"],
            "min_seg_size_forward": min(st["fwd_hdrs"], default=0),
            "Active Mean": float(np.mean(st["active"])) if st["active"] else 0.0,
            "Active Std": _sstd(st["active"]),
            "Active Max": max(st["active"], default=0.0),
            "Active Min": min(st["active"], default=0.0),
            "Idle Mean": float(np.mean(st["idle"])) if st["idle"] else 0.0,
            "Idle Std": _sstd(st["idle"]),
            "Idle Max": max(st["idle"], default=0.0),
            "Idle Min": min(st["idle"], default=0.0),
        }
    return out


def test_scapy_pcap_matches_cicflowmeter_all_71_features(tmp_path):
    """
    Constructs a multi-flow PCAP (TCP with options/windows/flags/idle-gap + UDP),
    runs Scapy's `extract_flows_from_pcap`, and verifies exact numerical parity on all
    71 CIC-IDS2017 features against the CICFlowMeter reference specification.
    """
    from scapy.all import Ether, IP, TCP, UDP, Raw, wrpcap, rdpcap

    pkts = []
    t0 = 1700000000.0

    src_mac = "00:11:22:33:44:55"
    dst_mac = "aa:bb:cc:dd:ee:ff"

    # --- Flow 1: Bidirectional TCP flow (with TCP options, CWE/ECE/URG/PSH, >5s idle burst) ---
    p1 = Ether(src=src_mac, dst=dst_mac) / IP(src="192.168.10.5", dst="10.20.30.40") / TCP(
        sport=50100, dport=443, flags="SEC", window=29200,
        options=[("MSS", 1460), ("NOP", None), ("WScale", 7)]
    )
    p1.time = t0
    p2 = Ether(src=dst_mac, dst=src_mac) / IP(src="10.20.30.40", dst="192.168.10.5") / TCP(
        sport=443, dport=50100, flags="SA", window=65535
    )
    p2.time = t0 + 0.010
    p3 = Ether(src=src_mac, dst=dst_mac) / IP(src="192.168.10.5", dst="10.20.30.40") / TCP(
        sport=50100, dport=443, flags="PA", window=29200
    ) / Raw(b"A" * 100)
    p3.time = t0 + 0.025
    # Idle gap of 6.500s (> 5.0s Active/Idle timeout & > 1.0s Subflow timeout)
    p4 = Ether(src=src_mac, dst=dst_mac) / IP(src="192.168.10.5", dst="10.20.30.40") / TCP(
        sport=50100, dport=443, flags="PAU", window=29200
    ) / Raw(b"B" * 300)
    p4.time = t0 + 6.525
    p5 = Ether(src=dst_mac, dst=src_mac) / IP(src="10.20.30.40", dst="192.168.10.5") / TCP(
        sport=443, dport=50100, flags="PA", window=64000
    ) / Raw(b"C" * 450)
    p5.time = t0 + 6.625
    p6 = Ether(src=src_mac, dst=dst_mac) / IP(src="192.168.10.5", dst="10.20.30.40") / TCP(
        sport=50100, dport=443, flags="FA", window=29200
    )
    p6.time = t0 + 6.700
    pkts.extend([p1, p2, p3, p4, p5, p6])

    # --- Flow 2: Bidirectional UDP DNS-like flow (verifies 8-byte header & -1 TCP window) ---
    u1 = Ether(src=src_mac, dst=dst_mac) / IP(src="192.168.10.5", dst="8.8.8.8") / UDP(sport=53001, dport=53) / Raw(b"Q" * 32)
    u1.time = t0 + 1.000
    u2 = Ether(src=src_mac, dst=dst_mac) / IP(src="192.168.10.5", dst="8.8.8.8") / UDP(sport=53001, dport=53) / Raw(b"Q" * 64)
    u2.time = t0 + 1.020
    u3 = Ether(src=dst_mac, dst=src_mac) / IP(src="8.8.8.8", dst="192.168.10.5") / UDP(sport=53, dport=53001) / Raw(b"R" * 128)
    u3.time = t0 + 1.050
    pkts.extend([u1, u2, u3])

    pcap_file = tmp_path / "cicflow_parity_test.pcap"
    wrpcap(str(pcap_file), pkts)

    decoded_pkts = rdpcap(str(pcap_file))
    ref_flows = _cicflowmeter_reference_from_packets(list(decoded_pkts))
    extracted_list = extract_flows_from_pcap(pcap_file)

    assert len(extracted_list) == 2
    by_key = {
        (f["source_ip"], f["destination_ip"], f["source_port"], f["Destination Port"], f["protocol"]): f
        for f in extracted_list
    }

    for flow_key, ref_feats in ref_flows.items():
        assert flow_key in by_key, f"Missing flow {flow_key} from PCAP extraction"
        actual = by_key[flow_key]
        for feat_name in config.CORE_FLOW_FEATURES:
            assert actual[feat_name] == pytest.approx(ref_feats[feat_name], rel=1e-6, abs=1e-6), (
                f"Feature mismatch for {flow_key} -> {feat_name}: actual={actual[feat_name]}, expected={ref_feats[feat_name]}"
            )

    # Sanity checks verifying old bugs stay fixed
    tcp_feat = by_key[("192.168.10.5", "10.20.30.40", 50100, 443, "TCP")]
    assert tcp_feat["Total Length of Fwd Packets"] == 400       # 0 + 100 + 300 payload bytes (not Ethernet frames)
    assert tcp_feat["act_data_pkt_fwd"] == 2                    # only p3 & p4 carry payload (not p1 SYN or p6 FIN)
    assert tcp_feat["Init_Win_bytes_forward"] == 29200          # from p1 SYN, not hardcoded 65535
    assert tcp_feat["Init_Win_bytes_backward"] == 65535         # from p2 SYN-ACK
    assert tcp_feat["Fwd Header Length"] == 28 + 20 + 20 + 20   # p1 has 28-byte TCP header w/ options
    assert tcp_feat["min_seg_size_forward"] == 20
    assert tcp_feat["Fwd PSH Flags"] == 2 and tcp_feat["Bwd PSH Flags"] == 1
    assert tcp_feat["CWE Flag Count"] == 1 and tcp_feat["ECE Flag Count"] == 1
    assert tcp_feat["Active Max"] == pytest.approx(25_000.0, rel=1e-4)
    assert tcp_feat["Idle Max"] == pytest.approx(6_500_000.0, rel=1e-4)

    udp_feat = by_key[("192.168.10.5", "8.8.8.8", 53001, 53, "UDP")]
    assert udp_feat["Fwd Header Length"] == 16
    assert udp_feat["Bwd Header Length"] == 8
    assert udp_feat["Init_Win_bytes_forward"] == -1
    assert udp_feat["Init_Win_bytes_backward"] == -1
