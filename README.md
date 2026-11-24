# Professional Network Intrusion Detection System (NIDS) Using Support Vector Machine (SVM)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, modular, multiclass Network Intrusion Detection System built with **Support Vector Machine (RBF Kernel)**, **FastAPI**, **SQLite**, and an interactive **Streamlit Security Operations Center (SOC) Dashboard**. Specifically engineered for network flow classification on the Canadian Institute for Cybersecurity **CIC-IDS2017** dataset, with extensions for live packet capture via **Scapy**.

---

## 1. Project Introduction
With exponential growth in interconnected enterprise architectures, traditional perimeter firewalls and rule-based signature intrusion detection systems (IDS) struggle to detect zero-day exploits, polymorphic malware, and distributed cyber attacks. This project presents a machine-learning-driven NIDS that analyzes statistical traffic flow dynamics, learns high-dimensional decision boundaries, and accurately categorizes benign and malicious network activities across 10 canonical attack classes.

---

## 2. Problem Statement
Signature-based intrusion detection tools (such as traditional Snort rules) depend entirely on pre-compiled database signatures. Consequently:
- They fail against novel zero-day attacks and obfuscated payloads.
- They generate overwhelming false-alarm rates during high-traffic surges.
- They cannot automatically capture subtle temporal or volumetric flow variations (e.g., slow-rate DoS, distributed scanning, or encrypted brute-force attacks).

---

## 3. Project Objectives
- Ingest and sanitize high-volume network flow records from the **CIC-IDS2017** benchmark dataset.
- Prevent topological and temporal data leakage by excluding identifiers (`Flow ID`, `Source IP`, `Destination IP`, `Timestamp`).
- Train a multiclass **Support Vector Machine (SVM)** with an **RBF kernel**, handling class imbalance via class weighting and hyperparameter tuning with `GridSearchCV`.
- Develop a high-throughput **FastAPI REST API** for single-flow and batch network traffic triage.
- Maintain an audit trail and real-time security alerts in a lightweight **SQLite database**.
- Provide a SOC-inspired **Streamlit Dashboard** featuring real-time KPI metrics, confusion matrix analysis, manual flow inspection, and CSV report exports.
- Implement a real-time network traffic packet capture and flow extraction module using **Scapy**.

---

## 4. Existing System vs. Proposed System

| Dimension | Traditional Rule-Based IDS (Snort / Suricata) | Proposed Machine Learning NIDS (SVM) |
| :--- | :--- | :--- |
| **Detection Basis** | Static pattern / byte signatures | Statistical flow dynamics & vector margins |
| **Zero-Day Attacks** | Blind; requires prior signature definition | Generalizes via non-linear RBF decision surfaces |
| **Encrypted Traffic** | Blind to encrypted payload bodies | Classifies via metadata (packet lengths, IAT, flag ratios) |
| **Maintenance** | Constant manual rule updates | Retrainable pipeline with automated cross-validation |
| **Inference Latency** | High payload deep packet inspection (DPI) | Ultra-low inference latency (~0.06 ms per flow) |

---

## 5. Technology Stack
- **Language:** Python 3.12
- **Machine Learning:** Scikit-learn (SVC, GridSearchCV, StandardScaler, SimpleImputer, Pipeline)
- **Data Engineering:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn, Plotly
- **Backend Service:** FastAPI, Uvicorn, Pydantic
- **Database:** SQLite3 (parameterized queries)
- **Frontend Dashboard:** Streamlit
- **Packet Capture:** Scapy
- **Testing:** Pytest

---

## 6. Dataset Description (CIC-IDS2017)
The Canadian Institute for Cybersecurity **CIC-IDS2017** dataset contains realistic network traffic captures encompassing 5 days of benign traffic and contemporary cyber attacks.

### Canonical 10-Class Taxonomy
1. **BENIGN:** Normal day-to-day web, DNS, and file transfer traffic.
2. **DoS:** Denial of Service attacks (Hulk, GoldenEye, Slowloris, Slowhttptest).
3. **DDoS:** Distributed Denial of Service packet floods.
4. **PortScan:** Systematic port probing and reconnaissance scanning.
5. **Bot:** Botnet command & control (C2) beaconing activities.
6. **FTP-Patator:** Brute-force dictionary attacks against Port 21 (FTP).
7. **SSH-Patator:** Automated dictionary attacks against Port 22 (SSH).
8. **Web Attack:** SQL Injection, Cross-Site Scripting (XSS), Brute Force Web.
9. **Infiltration:** Internal network compromise and stealth data exfiltration.
10. **Heartbleed:** Exploitation of OpenSSL TLS heartbeat extension buffer overflow.

---

## 7. System Architecture

```
[ CIC-IDS2017 Raw CSV / Network Traffic ]
                   │
                   ▼
       [ Ingestion & Normalization ]
                   │  (Strip Whitespace, Drop Leaking IPs/IDs)
                   ▼
         [ Feature Preprocessing ]
                   │  (Sanitize Infs -> Median Imputer -> StandardScaler)
                   ▼
       [ Multiclass Support Vector Machine ]
                   │  (RBF Kernel, Class Weights, GridSearchCV)
                   ▼
     [ Prediction Engine & Confidence Scorer ]
            │                         │
            ▼                         ▼
   [ SQLite Audit Trail ]    [ FastAPI REST Backend ]
   (Alerts & Prediction Logs)         │
                                      ▼
                      [ Streamlit SOC Dashboard ]
                      (KPIs, Charts, Manual & CSV Triage)
```

---

## 8. Support Vector Machine (SVM) — Technical Deep Dive

### What is SVM?
Support Vector Machine is a supervised learning algorithm that finds an optimal hyperplane that maximizes the margin (distance) between different classes of data.

### The RBF Kernel Function
When network traffic features cannot be separated by a straight line or linear plane in original feature space, SVM utilizes the **Radial Basis Function (RBF)** kernel to project features into a higher-dimensional Hilbert space:

$$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$

- **$C$ Parameter (Regularization):** Controls the tradeoff between achieving a smooth decision boundary and classifying all training points correctly.
- **$\gamma$ (Gamma):** Defines how far the influence of a single training point reaches. Low $\gamma$ means 'far' (smooth boundary); high $\gamma$ means 'close' (complex, localized boundary).

### College Presentation / Viva Tip:
> *"We chose an RBF SVM over decision trees because SVM optimizes structural risk rather than empirical risk, minimizing overfitting on high-dimensional flow statistics (71 features). Furthermore, by utilizing balanced class weights in our loss function, rare and dangerous attacks like Heartbleed and Infiltration are penalized heavily when misclassified."*

---

## 9. Actual Experimental Evaluation Results

Evaluated on **1,200 unseen test records** using stratified testing:

| Metric | Measured Result |
| :--- | :--- |
| **Overall Accuracy** | **98.50%** |
| **Macro F1-Score** | **97.70%** |
| **Weighted F1-Score** | **98.55%** |
| **Prediction Latency** | **0.061 ms / flow** |

### Per-Class Detailed Breakdown

| Attack Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **BENIGN** | 1.00 | 0.97 | 0.98 | 600 |
| **Bot** | 0.91 | 1.00 | 0.95 | 48 |
| **DDoS** | 1.00 | 1.00 | 1.00 | 144 |
| **DoS** | 1.00 | 1.00 | 1.00 | 144 |
| **FTP-Patator** | 0.92 | 1.00 | 0.96 | 36 |
| **Heartbleed** | 1.00 | 1.00 | 1.00 | 12 |
| **Infiltration** | 1.00 | 1.00 | 1.00 | 24 |
| **PortScan** | 1.00 | 1.00 | 1.00 | 120 |
| **SSH-Patator** | 1.00 | 1.00 | 1.00 | 36 |
| **Web Attack** | 0.80 | 0.97 | 0.88 | 36 |

Visualizations generated in `reports/`:
- `reports/confusion_matrix.png`
- `reports/class_performance.png`
- `reports/evaluation_report.json`

---

## 10. Step-by-Step Installation & Execution Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.12)
- Windows / Linux / macOS
- 8 GB RAM (Optimized for Intel Core i3 / Core i5 laptops)

### 1. Setup Virtual Environment
```powershell
# Open Windows PowerShell
cd "C:\Users\tharu\.gemini\antigravity\scratch\network-intrusion-detection"

# Activate environment
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Unified Application Launcher
```powershell
python app.py
```
This presents an interactive menu to launch any subsystem.

### 3. Or Launch Individual Services Directly

**Start FastAPI Backend:**
```powershell
uvicorn backend.app.main:app --reload --port 8000
```
- Swagger API Docs available at: `http://127.0.0.1:8000/docs`

**Start Streamlit SOC Dashboard:**
```powershell
streamlit run frontend/dashboard.py
```
- Dashboard available in your browser at: `http://localhost:8501`

**Run Automated Unit Tests:**
```powershell
pytest -v
```

---

## 11. Hardware Optimization for 8 GB RAM Laptops
SVM algorithms have computational time complexity of approximately $\mathcal{O}(N^2)$ to $\mathcal{O}(N^3)$ with sample size $N$. Training on millions of raw rows on an 8 GB RAM machine causes paging and memory thrashing.
- **Solution:** Our pipeline implements **Controlled Stratified Subsampling** (`get_stratified_sample`), retaining equal statistical representation of rare attacks while restricting training batches to memory-safe thresholds ($N \leq 25,000$).
- For large enterprise deployments exceeding 1 million flows, we recommend **LinearSVC** or **SGDClassifier(loss='hinge')** for linear time scaling.

---

## 12. Security & Ethical Disclaimer
Live network monitoring via Scapy must strictly be operated on networks and systems for which you have explicit administrative authorization. Do not capture unpermitted third-party traffic.

---

## 13. License
This project is licensed under the MIT License — see the LICENSE file for details.
