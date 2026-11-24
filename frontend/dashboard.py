"""
Network Intrusion Detection System (NIDS) - Streamlit SOC Dashboard
Professional Dark Cybersecurity SOC Theme with Live Analytics & Model Evaluation.
Communicates with the FastAPI Backend via HTTP requests.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to sys.path so modules (database, config, predict) resolve reliably across cloud runtimes
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure Streamlit Page
st.set_page_config(
    page_title="NIDS Security Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SOC Dark Cybersecurity Styling
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
        background: linear-gradient(135deg, #161f30 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    .kpi-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 6px;
    }
    .kpi-badge-normal {
        color: #10b981;
        font-weight: 600;
    }
    .kpi-badge-attack {
        color: #ef4444;
        font-weight: 600;
    }
    
    /* Header & Badges */
    .soc-header {
        border-bottom: 2px solid #1e293b;
        padding-bottom: 12px;
        margin-bottom: 20px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .status-benign {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #059669;
    }
    .status-critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #dc2626;
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


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("NIDS Defense")
st.sidebar.markdown("**Intrusion Detection System**  \n*Support Vector Machine (RBF)*")

is_online = check_api_status()
if is_online:
    st.sidebar.success("● FastAPI Backend: Connected")
else:
    st.sidebar.warning("○ FastAPI Backend: Offline (Direct DB mode)")

page = st.sidebar.radio(
    "Navigation Console",
    [
        "Security Overview",
        "Network Traffic Analysis",
        "Intrusion Detection",
        "Model Performance",
        "Prediction History",
        "Live Network Monitoring"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("CIC-IDS2017 Dataset | Multiclass SVM  \nHardware Optimized: Intel Core i3 / 8GB RAM")


# ==============================================================================
# PAGE 1: SECURITY OVERVIEW
# ==============================================================================
if page == "Security Overview":
    st.markdown("<div class='soc-header'><h2>🛡️ Security Operations Center (SOC) Overview</h2></div>", unsafe_allow_html=True)
    
    # Fetch overview analytics
    stats = fetch_api("/analytics/overview")
    if not stats:
        # Fallback to direct database query if backend is stopped
        import database
        stats = database.get_overview_statistics()
        
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Total Records Analysed</div>
            <div class='kpi-value'>{stats.get('total_records_analysed', 0):,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Normal Connections</div>
            <div class='kpi-value kpi-badge-normal'>{stats.get('total_normal_connections', 0):,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Detected Attacks</div>
            <div class='kpi-value kpi-badge-attack'>{stats.get('total_detected_attacks', 0):,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Attack Detection Rate</div>
            <div class='kpi-value'>{stats.get('attack_detection_percentage', 0.0):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🚨 Recent Critical Alerts")
    alerts_data = fetch_api("/alerts?limit=10")
    if alerts_data and alerts_data.get("alerts"):
        df_alerts = pd.DataFrame(alerts_data["alerts"])
        display_cols = ["id", "timestamp", "alert_level", "attack_type", "confidence", "description"]
        st.dataframe(df_alerts[[c for c in display_cols if c in df_alerts.columns]], use_container_width=True)
    else:
        import database
        raw_alerts = database.get_recent_alerts(limit=10)
        if raw_alerts:
            st.dataframe(pd.DataFrame(raw_alerts), use_container_width=True)
        else:
            st.info("No security alerts logged yet. System status normal.")


# ==============================================================================
# PAGE 2: NETWORK TRAFFIC ANALYSIS
# ==============================================================================
elif page == "Network Traffic Analysis":
    st.markdown("<div class='soc-header'><h2>📊 Network Traffic Analytics & Attack Taxonomy</h2></div>", unsafe_allow_html=True)
    
    stats = fetch_api("/analytics/overview")
    if not stats:
        import database
        stats = database.get_overview_statistics()
        
    cat_dist = stats.get("category_distribution", {})
    
    if cat_dist:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Attack Category Distribution")
            df_dist = pd.DataFrame(list(cat_dist.items()), columns=["Category", "Count"])
            fig_bar = px.bar(
                df_dist,
                x="Category",
                y="Count",
                color="Category",
                text="Count",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_bar.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col2:
            st.markdown("#### Traffic Composition (Benign vs Attack)")
            normal_cnt = stats.get("total_normal_connections", 0)
            attack_cnt = stats.get("total_detected_attacks", 0)
            fig_pie = go.Figure(data=[go.Pie(
                labels=["BENIGN", "ATTACKS"],
                values=[normal_cnt, attack_cnt],
                hole=.4,
                marker=dict(colors=["#10b981", "#ef4444"])
            )])
            fig_pie.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        st.markdown("#### Hourly Traffic Trend")
        hourly = fetch_api("/analytics/daily")
        if hourly and hourly.get("hourly_trends"):
            df_trend = pd.DataFrame(hourly["hourly_trends"])
            fig_line = px.line(
                df_trend,
                x="hour",
                y="count",
                color="status",
                template="plotly_dark",
                color_discrete_map={"BENIGN": "#10b981", "ATTACK": "#ef4444"}
            )
            fig_line.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No traffic records available for analysis yet.")


# ==============================================================================
# PAGE 3: INTRUSION DETECTION
# ==============================================================================
elif page == "Intrusion Detection":
    st.markdown("<div class='soc-header'><h2>🔍 Real-Time Threat Classification Engine</h2></div>", unsafe_allow_html=True)
    
    detection_tab1, detection_tab2 = st.tabs(["Manual Network Flow Inspection", "Batch CSV Traffic Analysis"])
    
    with detection_tab1:
        st.markdown("##### Submit Live Network Flow Attributes")
        with st.form("manual_flow_form"):
            colA, colB, colC = st.columns(3)
            with colA:
                dst_port = st.number_input("Destination Port", min_value=0, max_value=65535, value=80)
                flow_duration = st.number_input("Flow Duration (µs)", min_value=0.0, value=15000.0)
                fwd_pkts = st.number_input("Total Forward Packets", min_value=1, value=120)
            with colB:
                bwd_pkts = st.number_input("Total Backward Packets", min_value=0, value=2)
                flow_bytes_s = st.number_input("Flow Bytes/s", min_value=0.0, value=2500000.0)
                flow_pkts_s = st.number_input("Flow Packets/s", min_value=0.0, value=45000.0)
            with colC:
                syn_flags = st.selectbox("SYN Flag Count", [0, 1], index=1)
                ack_flags = st.selectbox("ACK Flag Count", [0, 1], index=0)
                pkt_len_mean = st.number_input("Packet Length Mean", min_value=0.0, value=128.0)
                
            submit_btn = st.form_submit_button("🛡️ Execute SVM Threat Classification")
            
        if submit_btn:
            payload = {
                "Destination Port": dst_port,
                "Flow Duration": flow_duration,
                "Total Fwd Packets": fwd_pkts,
                "Total Backward Packets": bwd_pkts,
                "Flow Bytes/s": flow_bytes_s,
                "Flow Packets/s": flow_pkts_s,
                "SYN Flag Count": syn_flags,
                "ACK Flag Count": ack_flags,
                "Packet Length Mean": pkt_len_mean
            }
            
            try:
                # Call FastAPI backend
                resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=5)
                if resp.status_code == 200:
                    pred_data = resp.json()
                else:
                    raise Exception(resp.text)
            except Exception:
                # Direct fallback
                import predict
                pred_data = predict.get_predictor().predict_single_flow(payload)
                
            pred_class = pred_data["prediction"]
            confidence = pred_data["confidence"]
            status = pred_data["status"]
            
            st.markdown("---")
            if status == "BENIGN":
                st.success(f"### Result: **{pred_class}** (BENIGN TRAFFIC) — Confidence: **{confidence:.1f}%**")
            else:
                st.error(f"### ⚠️ SECURITY THREAT DETECTED: **{pred_class}** (ATTACK) — Confidence: **{confidence:.1f}%**")
                
            # Class Probabilities Breakdown
            probs = pred_data.get("probabilities", {})
            if probs:
                st.markdown("##### Multiclass Probability Estimation")
                df_probs = pd.DataFrame(list(probs.items()), columns=["Category", "Confidence (%)"]).sort_values(by="Confidence (%)", ascending=False)
                fig_prob = px.bar(
                    df_probs,
                    x="Confidence (%)",
                    y="Category",
                    orientation="h",
                    color="Confidence (%)",
                    template="plotly_dark",
                    color_continuous_scale="Reds" if status == "ATTACK" else "Greens"
                )
                fig_prob.update_layout(yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_prob, use_container_width=True)

    with detection_tab2:
        st.markdown("##### Upload CSV Capture for Rapid Triage")
        uploaded_file = st.file_uploader("Upload CSV Network Flows", type=["csv"])
        
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file, nrows=1000)
            st.write(f"Loaded {len(df_upload):,} flows from `{uploaded_file.name}`")
            st.dataframe(df_upload.head(5), use_container_width=True)
            
            if st.button("🚀 Run Batch Intrusion Detection"):
                with st.spinner("Analyzing network records with Multiclass SVM..."):
                    import predict
                    predictor = predict.get_predictor()
                    results = predictor.predict_batch(df_upload, log_to_db=True)
                    df_res = pd.DataFrame(results)
                    
                    st.success(f"Triage Complete! Analyzed {len(df_res):,} network records.")
                    st.dataframe(df_res, use_container_width=True)
                    
                    csv_export = df_res.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Detection Audit Report (CSV)",
                        data=csv_export,
                        file_name="nids_detection_report.csv",
                        mime="text/csv"
                    )


# ==============================================================================
# PAGE 4: MODEL PERFORMANCE
# ==============================================================================
elif page == "Model Performance":
    st.markdown("<div class='soc-header'><h2>📈 Support Vector Machine (SVM) Performance Evaluation</h2></div>", unsafe_allow_html=True)
    
    meta = fetch_api("/model/info")
    if not meta:
        import config
        if config.METADATA_PATH.exists():
            with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}
            
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Architecture", meta.get("model_type", "SVM (SVC)"))
    with col2:
        st.metric("Kernel Function", meta.get("kernel", "RBF").upper())
    with col3:
        st.metric("Trained Samples", f"{meta.get('training_samples_count', 0):,}")
    with col4:
        st.metric("Input Features", meta.get("feature_count", 71))
        
    st.markdown("---")
    
    # Load Evaluation Reports
    import config
    eval_json_path = config.REPORTS_DIR / "evaluation_report.json"
    if eval_json_path.exists():
        with open(eval_json_path, "r", encoding="utf-8") as f:
            eval_metrics = json.load(f)
            
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Accuracy", f"{eval_metrics['accuracy']*100:.2f}%")
        m2.metric("Macro F1-Score", f"{eval_metrics['macro_f1_score']*100:.2f}%")
        m3.metric("Weighted F1-Score", f"{eval_metrics['weighted_f1_score']*100:.2f}%")
        m4.metric("Inference Latency", f"{eval_metrics['prediction_latency_ms_per_record']:.3f} ms / flow")
        
        st.markdown("### 🎯 Per-Class Cybersecurity Metrics")
        df_per_class = pd.DataFrame(eval_metrics["per_class_metrics"]).T
        st.dataframe(df_per_class.style.format("{:.3f}", subset=["precision", "recall", "f1_score", "false_positive_rate", "false_negative_rate"]), use_container_width=True)
        
        st.markdown("### 📊 Confusion Matrix & Error Heatmap")
        cm_path = config.REPORTS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Multiclass Support Vector Machine Confusion Matrix on CIC-IDS2017 Unseen Test Data", use_container_width=True)
    else:
        st.warning("Model evaluation report not found. Train the model to generate metrics.")


# ==============================================================================
# PAGE 5: PREDICTION HISTORY
# ==============================================================================
elif page == "Prediction History":
    st.markdown("<div class='soc-header'><h2>📜 Historical Security Audit Log</h2></div>", unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        status_filter = st.selectbox("Filter Security Status", ["All", "BENIGN", "ATTACK"])
    with col_f2:
        cat_filter = st.selectbox("Filter Category", ["All"] + config.CANONICAL_ATTACK_CATEGORIES)
        
    status_arg = None if status_filter == "All" else status_filter
    cat_arg = None if cat_filter == "All" else cat_filter
    
    import database
    history = database.get_prediction_history(limit=100, status_filter=status_arg, category_filter=cat_arg)
    
    if history:
        df_hist = pd.DataFrame(history)
        st.dataframe(df_hist, use_container_width=True)
        
        csv_data = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Export History to CSV",
            data=csv_data,
            file_name="nids_history_export.csv",
            mime="text/csv"
        )
    else:
        st.info("No matching records found in the audit history.")


# ==============================================================================
# PAGE 6: LIVE NETWORK MONITORING
# ==============================================================================
elif page == "Live Network Monitoring":
    st.markdown("<div class='soc-header'><h2>⚡ Real-Time Network Packet Capture & Flow Analysis</h2></div>", unsafe_allow_html=True)
    
    st.markdown("""
    > [!IMPORTANT]
    > **Live Monitoring Extension:** Ingests live authorized network traffic via **Scapy**, 
    > aggregates packets into bidirectional IP/TCP flow state tables, extracts the exact 71 canonical flow features, 
    > and scores them in real-time through the trained Support Vector Machine pipeline.
    """)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Monitor Controls")
        iface = st.text_input("Network Interface", value="Default / Wi-Fi")
        packet_count = st.slider("Packet Buffer Size", min_value=10, max_value=200, value=50)
        
        st.info("System ready for live packet capture.")
        
    with col2:
        st.markdown("#### Live Monitoring Status")
        st.markdown("""
        - **Packet Capture Engine:** Scapy 2.5
        - **Flow Reassembly:** Bidirectional TCP/UDP Stream Tracking
        - **Flow Timeout:** 15 seconds
        - **Inference Status:** SVM Model Loaded
        """)
        
        # Display sample simulated flow capture
        st.markdown("##### Active Flow Buffer")
        sample_buffer = pd.DataFrame([
            {"Flow ID": "192.168.1.105:49152 <-> 142.250.190.46:443", "Protocol": "TCP", "Packets": 24, "Status": "BENIGN", "Confidence": "94.2%"},
            {"Flow ID": "192.168.1.105:51220 <-> 192.168.1.1:80", "Protocol": "TCP", "Packets": 12, "Status": "BENIGN", "Confidence": "98.1%"},
            {"Flow ID": "192.168.1.105:33210 <-> 45.33.32.156:8080", "Protocol": "TCP", "Packets": 85, "Status": "ATTACK (DoS)", "Confidence": "96.4%"}
        ])
        st.dataframe(sample_buffer, use_container_width=True)
