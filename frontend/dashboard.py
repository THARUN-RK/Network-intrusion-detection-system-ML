"""
Network Intrusion Detection System (NIDS) - Streamlit Security Operations Center (SOC)
Professional Cyber Defense Dashboard with Real-Time Threat Feeds, IP Reputation,
Incident Investigation, Severity Indicators, and PDF/CSV Audit Reports.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to sys.path so modules resolve reliably across all cloud runtimes
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import database
import predict

import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
st.set_option('dataframe.showArrow', False)

# Configure Streamlit Page
st.set_page_config(
    page_title="NIDS Security Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SOC Dark Cybersecurity Theme
st.markdown("""
<style>
    /* Dark SOC Theme Overrides */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* KPI Card styling */
    .kpi-card {
        background: linear-gradient(135deg, #131b2e 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
        margin-bottom: 12px;
    }
    .kpi-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    .kpi-badge-normal { color: #10b981; }
    .kpi-badge-attack { color: #ef4444; }
    .kpi-badge-critical { color: #f43f5e; font-weight: bold; }
    .kpi-badge-anomaly { color: #f59e0b; font-weight: bold; }
    
    /* Header & Badges */
    .soc-header {
        border-bottom: 2px solid #1e293b;
        padding-bottom: 10px;
        margin-bottom: 18px;
    }
    
    /* Severity Badges */
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #dc2626;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.75rem;
    }
    .badge-high {
        background-color: rgba(249, 115, 22, 0.2);
        color: #fb923c;
        border: 1px solid #ea580c;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.75rem;
    }
    .badge-medium {
        background-color: rgba(234, 179, 8, 0.2);
        color: #facc15;
        border: 1px solid #ca8a04;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.75rem;
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #059669;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

API_BASE_URL = os.environ.get("NIDS_API_URL", "http://127.0.0.1:8000")


def check_api_status() -> bool:
    """Checks if FastAPI backend is reachable."""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return r.status_code == 200 and r.json().get("status") == "healthy"
    except Exception:
        return False


def fetch_api(endpoint: str, default=None):
    """Safe HTTP GET wrapper."""
    try:
        r = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
        if r.status_code == 200:
            return r.json()
        return default
    except Exception:
        return default


# ==============================================================================
# SIDEBAR NAVIGATION & AUTHENTICATION
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=60)
st.sidebar.title("NIDS Defense")
st.sidebar.markdown("**Security Operations Center (SOC)**  \n*Multiclass SVM (RBF Kernel)*")

is_online = check_api_status()
if is_online:
    st.sidebar.success("● FastAPI Backend: Connected")
else:
    st.sidebar.info("○ Mode: Standalone Direct Database")

# Auth State Simulation / Management
if "auth_token" not in st.session_state:
    st.session_state["auth_token"] = None
    st.session_state["username"] = "analyst"
    st.session_state["role"] = "analyst"

with st.sidebar.expander("👤 SOC Operator Session", expanded=False):
    st.caption(f"Logged in as: **{st.session_state['username']}** ({st.session_state['role'].upper()})")
    u_input = st.text_input("Username", value="admin")
    p_input = st.text_input("Password", type="password", value="Admin@12345")
    if st.button("Authenticate (JWT)"):
        try:
            resp = requests.post(f"{API_BASE_URL}/auth/login", json={"username": u_input, "password": p_input}, timeout=3)
            if resp.status_code == 200:
                t_data = resp.json()
                st.session_state["auth_token"] = t_data["access_token"]
                st.session_state["username"] = t_data["username"]
                st.session_state["role"] = t_data["role"]
                st.success(f"Authenticated as {t_data['role'].upper()}!")
            else:
                st.error("Invalid credentials.")
        except Exception:
            # Fallback auth for offline demo
            if u_input == "admin" and p_input == "Admin@12345":
                st.session_state["username"] = "admin"
                st.session_state["role"] = "admin"
                st.success("Authenticated as ADMIN!")
            else:
                st.error("Authentication failed.")

page = st.sidebar.radio(
    "Operations Console",
    [
        "Security Overview",
        "Threat Analytics & Timeline",
        "Intrusion Detection Engine",
        "IP Monitoring & Reputation",
        "Incident Investigation",
        "Model Performance & Metrics",
        "Executive Reports & Audit Logs",
        "Live Network Monitoring"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("CIC-IDS2017 Dataset | Multiclass SVM  \nMulti-DB: SQLite & PostgreSQL Ready")


# ==============================================================================
# PAGE 1: SECURITY OVERVIEW
# ==============================================================================
if page == "Security Overview":
    st.markdown("<div class='soc-header'><h2>🛡️ Security Operations Center (SOC) Overview</h2></div>", unsafe_allow_html=True)
    
    stats = fetch_api("/analytics/overview")
    if not stats:
        stats = database.get_overview_statistics()
        
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Total Analyzed Flows</div>
            <div class='kpi-value'>{stats.get('total_records_analysed', 0):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Normal Connections</div>
            <div class='kpi-value kpi-badge-normal'>{stats.get('total_normal_connections', 0):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Detected Attacks</div>
            <div class='kpi-value kpi-badge-attack'>{stats.get('total_detected_attacks', 0):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Zero-Day Anomalies</div>
            <div class='kpi-value kpi-badge-anomaly'>{stats.get('total_anomalies_detected', 0):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Attack Ratio</div>
            <div class='kpi-value'>{stats.get('attack_detection_percentage', 0.0):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🚨 Live Critical Security Alerts")
    alerts_data = fetch_api("/alerts?limit=10")
    alerts_list = alerts_data.get("alerts", []) if alerts_data else database.get_recent_alerts(limit=10)
    
    if alerts_list:
        df_alerts = pd.DataFrame(alerts_list)
        st.dataframe(df_alerts, use_container_width=True)
        
        # Interactive alert resolution actions
        st.markdown("##### Alert Triage Actions")
        act_col1, act_col2, act_col3 = st.columns([1, 1, 2])
        with act_col1:
            sel_alert_id = st.selectbox("Select Alert ID", [a["id"] for a in alerts_list])
        with act_col2:
            new_st = st.selectbox("Set Triage Status", ["INVESTIGATING", "RESOLVED", "FALSE_POSITIVE"])
        with act_col3:
            st.write("")
            st.write("")
            if st.button("Apply Status Update"):
                try:
                    requests.patch(f"{API_BASE_URL}/soc/alerts/{sel_alert_id}/status?status={new_st}", timeout=2)
                except Exception:
                    database.update_alert_status(sel_alert_id, new_st)
                st.success(f"Alert {sel_alert_id} updated to {new_st}!")
                st.rerun()
    else:
        st.info("System healthy. No active security alerts.")


# ==============================================================================
# PAGE 2: THREAT ANALYTICS & TIMELINE
# ==============================================================================
elif page == "Threat Analytics & Timeline":
    st.markdown("<div class='soc-header'><h2>📊 Threat Intelligence, Taxonomy & Timeline</h2></div>", unsafe_allow_html=True)
    
    stats = fetch_api("/analytics/overview") or database.get_overview_statistics()
    cat_dist = stats.get("category_distribution", {})
    sev_dist = stats.get("severity_distribution", {})
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("#### Attack Category Distribution")
        if cat_dist:
            df_cat = pd.DataFrame(list(cat_dist.items()), columns=["Category", "Count"])
            fig_bar = px.bar(
                df_cat, x="Category", y="Count", color="Category",
                template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_bar.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)
            
    with col_chart2:
        st.markdown("#### Threat Severity Breakdown")
        if sev_dist:
            fig_donut = go.Figure(data=[go.Pie(
                labels=list(sev_dist.keys()),
                values=list(sev_dist.values()),
                hole=0.45,
                marker=dict(colors=["#10b981", "#f59e0b", "#f97316", "#ef4444"])
            )])
            fig_donut.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_donut, use_container_width=True)
            
    st.markdown("#### Hourly Incident Activity Trend")
    hourly = fetch_api("/analytics/daily")
    if hourly and hourly.get("hourly_trends"):
        df_hourly = pd.DataFrame(hourly["hourly_trends"])
        fig_time = px.area(
            df_hourly, x="hour", y="count", color="status",
            template="plotly_dark", color_discrete_map={"BENIGN": "#10b981", "ATTACK": "#ef4444"}
        )
        fig_time.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_time, use_container_width=True)


# ==============================================================================
# PAGE 3: INTRUSION DETECTION ENGINE
# ==============================================================================
elif page == "Intrusion Detection Engine":
    st.markdown("<div class='soc-header'><h2>🔍 Real-Time Threat Classification Engine</h2></div>", unsafe_allow_html=True)
    
    det_tab1, det_tab2 = st.tabs(["Single Flow Inspector", "Batch CSV Traffic Triage"])
    
    with det_tab1:
        with st.form("manual_flow_form"):
            r1, r2, r3 = st.columns(3)
            with r1:
                src_ip = st.text_input("Source IP", value="192.168.1.105")
                dst_port = st.number_input("Destination Port", min_value=0, max_value=65535, value=80)
                flow_duration = st.number_input("Flow Duration (µs)", min_value=0.0, value=15000.0)
            with r2:
                dst_ip = st.text_input("Destination IP", value="10.0.0.1")
                fwd_pkts = st.number_input("Total Forward Packets", min_value=1, value=120)
                bwd_pkts = st.number_input("Total Backward Packets", min_value=0, value=2)
            with r3:
                proto = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"])
                flow_bytes_s = st.number_input("Flow Bytes/s", min_value=0.0, value=2500000.0)
                flow_pkts_s = st.number_input("Flow Packets/s", min_value=0.0, value=45000.0)
                
            r4, r5 = st.columns(2)
            with r4:
                syn_flags = st.selectbox("SYN Flag Count", [0, 1], index=1)
            with r5:
                ack_flags = st.selectbox("ACK Flag Count", [0, 1], index=0)
                
            btn_classify = st.form_submit_button("🛡️ Run SVM Multiclass Classification")
            
        if btn_classify:
            payload = {
                "Source IP": src_ip,
                "Destination IP": dst_ip,
                "Destination Port": dst_port,
                "Protocol": proto,
                "Flow Duration": flow_duration,
                "Total Fwd Packets": fwd_pkts,
                "Total Backward Packets": bwd_pkts,
                "Flow Bytes/s": flow_bytes_s,
                "Flow Packets/s": flow_pkts_s,
                "SYN Flag Count": syn_flags,
                "ACK Flag Count": ack_flags
            }
            try:
                resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=4)
                pred_data = resp.json() if resp.status_code == 200 else predict.get_predictor().predict_single_flow(payload)
            except Exception:
                pred_data = predict.get_predictor().predict_single_flow(payload)
                
            p_class = pred_data["prediction"]
            conf = pred_data["confidence"]
            status = pred_data["status"]
            sev = pred_data.get("severity", "LOW")
            is_ano = pred_data.get("is_anomaly", False)
            
            st.markdown("---")
            if is_ano:
                st.warning(f"### ⚠️ ZERO-DAY ANOMALY DETECTED: **{p_class}** — Severity: **{sev}**")
            elif status == "BENIGN":
                st.success(f"### ✓ Normal Network Traffic: **{p_class}** — Confidence: **{conf:.1f}%**")
            else:
                st.error(f"### 🚨 CYBERATTACK IDENTIFIED: **{p_class}** — Severity: **{sev}** ({conf:.1f}% confidence)")
                
            probs = pred_data.get("probabilities", {})
            if probs:
                df_probs = pd.DataFrame(list(probs.items()), columns=["Category", "Confidence (%)"]).sort_values(by="Confidence (%)", ascending=False)
                fig_p = px.bar(df_probs, x="Confidence (%)", y="Category", orientation="h", color="Confidence (%)", template="plotly_dark")
                fig_p.update_layout(yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_p, use_container_width=True)

    with det_tab2:
        st.markdown("##### Upload High-Volume Traffic Capture (CSV)")
        up_file = st.file_uploader("Select Capture File", type=["csv"])
        if up_file:
            df_in = pd.read_csv(up_file, nrows=1000)
            st.write(f"Loaded {len(df_in):,} flows from `{up_file.name}`")
            if st.button("🚀 Process Batch Flows"):
                with st.spinner("Analyzing with Multiclass SVM..."):
                    results = predict.get_predictor().predict_batch(df_in, log_to_db=True)
                    df_res = pd.DataFrame(results)
                    st.success(f"Triage complete! {len(df_res):,} records classified.")
                    st.dataframe(df_res, use_container_width=True)


# ==============================================================================
# PAGE 4: IP MONITORING & REPUTATION
# ==============================================================================
elif page == "IP Monitoring & Reputation":
    st.markdown("<div class='soc-header'><h2>🌐 IP Threat Monitoring & Reputation Profiler</h2></div>", unsafe_allow_html=True)
    
    flt_col1, flt_col2 = st.columns([1, 2])
    with flt_col1:
        threat_filter = st.selectbox("Filter Threat Level", ["All", "SAFE", "SUSPICIOUS", "MALICIOUS"])
        
    ip_profiles = fetch_api(f"/soc/ips?threat_level={threat_filter}") or database.get_tracked_ips(threat_filter=threat_filter)
    
    if ip_profiles:
        df_ips = pd.DataFrame(ip_profiles)
        st.dataframe(df_ips, use_container_width=True)
        
        st.markdown("##### Host Defense & Firewall Control")
        b_col1, b_col2, b_col3 = st.columns([2, 1, 1])
        with b_col1:
            target_ip = st.selectbox("Select IP Address", [ip["ip_address"] for ip in ip_profiles])
        with b_col2:
            st.write("")
            st.write("")
            if st.button("🚫 Block IP Address"):
                try:
                    requests.post(f"{API_BASE_URL}/soc/ips/block", json={"ip_address": target_ip, "block": True}, timeout=2)
                except Exception:
                    database.toggle_ip_block(target_ip, True)
                st.warning(f"Firewall rule enacted: {target_ip} is now BLOCKED.")
                st.rerun()
        with b_col3:
            st.write("")
            st.write("")
            if st.button("✓ Unblock IP Address"):
                try:
                    requests.post(f"{API_BASE_URL}/soc/ips/block", json={"ip_address": target_ip, "block": False}, timeout=2)
                except Exception:
                    database.toggle_ip_block(target_ip, False)
                st.success(f"{target_ip} has been unblocked.")
                st.rerun()
    else:
        st.info("No IP addresses profiled under this filter yet.")


# ==============================================================================
# PAGE 5: INCIDENT INVESTIGATION
# ==============================================================================
elif page == "Incident Investigation":
    st.markdown("<div class='soc-header'><h2>📑 SOC Incident Response & Investigation Workspace</h2></div>", unsafe_allow_html=True)
    
    inc_tab1, inc_tab2 = st.tabs(["Active Incidents", "Create New Incident Ticket"])
    
    with inc_tab1:
        st.markdown("##### Ongoing Incident Investigations")
        incidents = fetch_api("/soc/incidents") or database.get_incidents()
        if incidents:
            df_inc = pd.DataFrame(incidents)
            st.dataframe(df_inc, use_container_width=True)
            
            st.markdown("##### Update Incident Investigation")
            u_col1, u_col2, u_col3 = st.columns([1, 1, 2])
            with u_col1:
                sel_inc_code = st.selectbox("Select Incident Code", [i["incident_code"] for i in incidents])
            with u_col2:
                new_inc_status = st.selectbox("Status", ["OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"])
            with u_col3:
                note_text = st.text_input("Investigator Note", placeholder="e.g. Host quarantined; IP blacklisted on edge gateway.")
                
            if st.button("Save Incident Investigation Note"):
                try:
                    requests.patch(f"{API_BASE_URL}/soc/incidents/{sel_inc_code}", json={"status": new_inc_status, "notes": note_text}, timeout=2)
                except Exception:
                    database.update_incident_status(sel_inc_code, new_inc_status, note_text)
                st.success(f"Incident {sel_inc_code} updated successfully!")
                st.rerun()
        else:
            st.info("No active incidents found.")

    with inc_tab2:
        st.markdown("##### File a Formal Incident Report")
        with st.form("new_incident_form"):
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                inc_title = st.text_input("Incident Title", value="Volumetric SYN Flood on Port 80")
                inc_sev = st.selectbox("Incident Severity", ["LOW", "MEDIUM", "HIGH", "CRITICAL"], index=2)
                inc_src = st.text_input("Source Host / IP", value="192.168.1.105")
            with t_col2:
                inc_tgt = st.text_input("Target Asset / IP", value="10.0.0.1:80")
                inc_assignee = st.text_input("Assigned Analyst", value=st.session_state.get("username", "analyst"))
                inc_desc = st.text_area("Detailed Incident Summary", value="Identified high-velocity connection requests exceeding 25,000 pkts/s.")
                
            create_btn = st.form_submit_button("🚨 Submit Incident to Queue")
            if create_btn:
                payload = {
                    "title": inc_title,
                    "description": inc_desc,
                    "severity": inc_sev,
                    "source_ip": inc_src,
                    "target_ip": inc_tgt,
                    "assigned_to": inc_assignee
                }
                try:
                    requests.post(f"{API_BASE_URL}/soc/incidents", json=payload, timeout=3)
                except Exception:
                    database.create_incident(**payload)
                st.success("New incident created successfully!")
                st.rerun()


# ==============================================================================
# PAGE 6: MODEL PERFORMANCE & METRICS
# ==============================================================================
elif page == "Model Performance & Metrics":
    st.markdown("<div class='soc-header'><h2>📈 SVM RBF Performance & Calibration Report</h2></div>", unsafe_allow_html=True)
    
    meta = fetch_api("/model/info") or (json.load(open(config.METADATA_PATH)) if config.METADATA_PATH.exists() else {})
    
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Model Architecture", meta.get("model_type", "SVM (SVC)"))
    with m2: st.metric("Kernel Function", meta.get("kernel", "RBF").upper())
    with m3: st.metric("Training Samples", f"{meta.get('training_samples_count', 0):,}")
    with m4: st.metric("Input Features", meta.get("feature_count", 71))
    
    st.markdown("---")
    tab_eval, tab_daywise, tab_calib = st.tabs([
        "🎯 Multiclass Evaluation & Matrix",
        "📅 Day-Wise Temporal Generalization",
        "⚖️ Confidence Calibration & Anomaly Rejection"
    ])
    
    with tab_eval:
        eval_json = config.REPORTS_DIR / "evaluation_report.json"
        if eval_json.exists():
            eval_data = json.load(open(eval_json))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall Accuracy", f"{eval_data['accuracy']*100:.2f}%")
            c2.metric("Macro F1-Score", f"{eval_data['macro_f1_score']*100:.2f}%")
            c3.metric("Weighted F1-Score", f"{eval_data['weighted_f1_score']*100:.2f}%")
            c4.metric("Inference Latency", f"{eval_data['prediction_latency_ms_per_record']:.3f} ms / flow")
            
            st.markdown("### 🎯 Per-Class Precision, Recall & Error Rates")
            df_per = pd.DataFrame(eval_data["per_class_metrics"]).T
            fmt_cols = [c for c in ["precision", "recall", "f1_score", "false_positive_rate", "false_negative_rate"] if c in df_per.columns]
            if fmt_cols:
                df_per[fmt_cols] = df_per[fmt_cols].apply(lambda x: x.map("{:.3f}".format))
            st.dataframe(df_per, use_container_width=True)
            
            col_cm1, col_cm2 = st.columns(2)
            with col_cm1:
                cm_path = config.REPORTS_DIR / "confusion_matrix.png"
                if cm_path.exists():
                    st.image(str(cm_path), caption="Multiclass Support Vector Machine Confusion Matrix (CIC-IDS2017)", use_container_width=True)
            with col_cm2:
                perf_path = config.REPORTS_DIR / "class_performance.png"
                if perf_path.exists():
                    st.image(str(perf_path), caption="Class-wise Precision, Recall, and F1-Score", use_container_width=True)
        else:
            st.info("Evaluation metrics report not found. Run evaluation.py to generate.")

    with tab_daywise:
        st.markdown("### 📅 Day-Wise Cross-Testing (CIC-IDS2017 Temporal Schedule)")
        st.markdown("""
        In real-world cybersecurity, attack vectors vary dramatically by day. 
        Testing the RBF SVM model across simulated Monday–Friday attack profiles proves that the model learns 
        **actual protocol behavior and volumetric dynamics**, rather than memorizing temporal timestamps or ephemeral IP artifacts.
        """)
        
        day_report_path = config.REPORTS_DIR / "day_wise_report.json"
        if day_report_path.exists():
            day_data = json.load(open(day_report_path))
            rows = []
            for d_name, d_val in day_data.get("days", {}).items():
                rows.append({
                    "Day": d_name,
                    "Attack Scenario": d_val.get("description", ""),
                    "Total Flows": f"{d_val.get('total_records', 0):,}",
                    "Attacks": f"{d_val.get('attack_records', 0):,}",
                    "Accuracy": f"{d_val.get('accuracy', 0)*100:.2f}%",
                    "Macro F1": f"{d_val.get('macro_f1', 0)*100:.2f}%",
                    "Attack Detection": f"{d_val.get('attack_detection_rate', 0)*100:.2f}%",
                    "Benign FPR": f"{d_val.get('false_positive_rate', 0)*100:.2f}%",
                    "Anomaly Rejections": d_val.get("anomaly_rejections", 0),
                    "Latency (ms)": f"{d_val.get('inference_latency_ms', 0):.3f}"
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            
            day_img_path = config.REPORTS_DIR / "day_wise_evaluation.png"
            if day_img_path.exists():
                st.image(str(day_img_path), caption="Day-Wise Temporal Generalization, FPR, and Latency Across Monday–Friday", use_container_width=True)
        else:
            if st.button("🚀 Run Day-Wise Cross-Testing Benchmark"):
                with st.spinner("Executing temporal evaluation on 15,000 flows..."):
                    import day_wise_evaluation
                    day_wise_evaluation.evaluate_day_wise()
                st.success("Day-wise cross-testing complete!")
                st.rerun()

    with tab_calib:
        st.markdown("### ⚖️ Confidence Calibration & Zero-Day Anomaly Rejection")
        st.markdown("""
        **RBF Kernel Non-Linear Decision Geometry:**
        - Standard linear models fail against polymorphic malware and distributed attacks. 
        - The Radial Basis Function (RBF) kernel projects 71-dimensional network flow vectors into an infinite-dimensional reproducing kernel Hilbert space (RKHS).
        - Posterior class probabilities are calibrated using Platt scaling (sigmoid transformation over SVM decision function values).
        """)
        
        c_cal1, c_cal2 = st.columns(2)
        with c_cal1:
            st.metric("Anomaly Confidence Threshold", "55.0%", help="Flows whose highest class probability falls below 55% are flagged as UNKNOWN / ANOMALY")
        with c_cal2:
            st.metric("Rejection Action", "Flagged as Zero-Day / Anomaly", help="Alerts SOC analysts to zero-day novel patterns")
            
        st.info("💡 **Operational Benefit:** Known benign and high-certainty attacks are classified directly with high confidence. When polymorphic or zero-day network flows deviate from known support vectors, the confidence score drops below 55.0%, preventing false positives and triggering priority SOC investigation.")


# ==============================================================================
# PAGE 7: EXECUTIVE REPORTS & AUDIT LOGS
# ==============================================================================
elif page == "Executive Reports & Audit Logs":
    st.markdown("<div class='soc-header'><h2>📜 Executive Security Reports & Audit Export</h2></div>", unsafe_allow_html=True)
    
    rep_col1, rep_col2 = st.columns(2)
    with rep_col1:
        st.markdown("#### 📄 Executive PDF Audit Report")
        st.markdown("Generates a formal, printable PDF document compiling executive traffic statistics, active critical alerts, and high-threat IP profiles.")
        
        # Download button calling the backend PDF generator
        pdf_url = f"{API_BASE_URL}/soc/reports/pdf"
        st.markdown(f"""
        <a href='{pdf_url}' target='_blank' style='text-decoration:none;'>
            <button style='background-color:#0284c7; color:white; padding:12px 24px; border:none; border-radius:6px; font-weight:700; cursor:pointer;'>
                📥 Download Executive PDF Audit Report
            </button>
        </a>
        """, unsafe_allow_html=True)
        
    with rep_col2:
        st.markdown("#### 📊 CSV Historical Audit Trail")
        history = database.get_prediction_history(limit=500)
        if history:
            df_hist = pd.DataFrame(history)
            st.download_button(
                "📥 Export Audit History (CSV)",
                data=df_hist.to_csv(index=False).encode('utf-8'),
                file_name="nids_audit_history.csv",
                mime="text/csv"
            )
            
    st.markdown("---")
    st.markdown("### Historical Audit Trail")
    history_records = database.get_prediction_history(limit=100)
    if history_records:
        st.dataframe(pd.DataFrame(history_records), use_container_width=True)


# ==============================================================================
# PAGE 8: LIVE NETWORK MONITORING
# ==============================================================================
elif page == "Live Network Monitoring":
    st.markdown("<div class='soc-header'><h2>⚡ Real-Time Network Packet Capture & Flow Analysis</h2></div>", unsafe_allow_html=True)
    
    st.info("Scapy packet capture engine is ready. Bidirectional TCP/UDP stream reassembly maps packets directly to the 71-feature SVM schema.")
    
    col_l1, col_l2 = st.columns([1, 2])
    with col_l1:
        st.markdown("#### Monitor Controls")
        iface = st.text_input("Network Adapter", value="Default / Wi-Fi")
        pkt_cnt = st.slider("Flow Extraction Batch Size", min_value=10, max_value=200, value=50)
        st.markdown("""
        - **Packet Capture:** Scapy 2.5
        - **Stream Tracking:** Bidirectional 5-Tuple
        - **Inactivity Timeout:** 5.0 seconds
        - **Anomaly Threshold:** 55.0%
        """)
        
    with col_l2:
        st.markdown("#### Active Flow Buffer & Real-Time Threat Stream")
        recent_preds = database.get_prediction_history(limit=15)
        if recent_preds:
            st.dataframe(pd.DataFrame(recent_preds)[["id", "timestamp", "source_ip", "destination_ip", "destination_port", "predicted_category", "severity", "confidence"]], use_container_width=True)
        else:
            st.info("Awaiting live network packets...")
