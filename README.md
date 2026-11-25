# 🛡️ Enterprise Network Intrusion Detection System (NIDS) & Security Operations Center (SOC)

[![Live SOC Platform](https://img.shields.io/badge/Live%20Platform-Render-0284c7?style=for-the-badge&logo=render)](https://network-intrusion-detection-system-ml.onrender.com/)
[![GitHub Pages](https://img.shields.io/badge/Web%20Portal-GitHub%20Pages-22c55e?style=for-the-badge&logo=github)](https://tharun-rk.github.io/Network-intrusion-detection-system-ML/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/SOC%20Console-Streamlit-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Container-Docker%20Compose-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-20%20Passed-brightgreen.svg)](tests/)

An enterprise-grade, full-stack **Network Intrusion Detection System (NIDS)** and interactive **Security Operations Center (SOC)**. Built with **Multiclass Support Vector Machine (RBF Kernel)**, **FastAPI REST API**, **WebSockets**, **SQLAlchemy (SQLite + PostgreSQL)**, **Scapy Live Packet Capture**, **ReportLab Executive PDF Generator**, and an 8-console **Streamlit SOC Dashboard**. 

Trained and evaluated on the Canadian Institute for Cybersecurity **CIC-IDS2017** dataset across 71 network flow features, with zero-day anomaly rejection and day-wise temporal cross-validation.

---

## 🌐 Live Deployments & Quick Access

- **Live SOC Production Platform:** [network-intrusion-detection-system-ml.onrender.com](https://network-intrusion-detection-system-ml.onrender.com/)
- **GitHub Pages Portal:** [tharun-rk.github.io/Network-intrusion-detection-system-ML](https://tharun-rk.github.io/Network-intrusion-detection-system-ML/)
- **Swagger REST API Documentation:** `https://network-intrusion-detection-system-ml.onrender.com/docs`
- **Default SOC Credentials:**
  - **Admin Role:** Username: `admin` | Password: `Admin@12345`
  - **Analyst Role:** Username: `analyst` | Password: `Analyst@12345`

---

## 📑 Table of Contents
1. [Key Features](#-key-features)
2. [System Architecture](#-system-architecture)
3. [SOC Operational Consoles (8 Modules)](#-soc-operational-consoles)
4. [Dataset & Canonical Attack Taxonomy](#-dataset--canonical-attack-taxonomy)
5. [Machine Learning Engine (SVM Deep Dive)](#-machine-learning-engine)
6. [Day-Wise Temporal Generalization Results](#-day-wise-temporal-generalization-results)
7. [API Specification & Real-Time WebSockets](#-api-specification--real-time-websockets)
8. [Multi-Database Architecture](#-multi-database-architecture)
9. [Installation & Execution Guide](#-installation--execution-guide)
10. [Docker Container Deployment](#-docker-container-deployment)
11. [Automated Testing Suite (Pytest)](#-automated-testing-suite)
12. [Hardware Optimization & CSE Presentation Guide](#-hardware-optimization--cse-presentation-guide)

---

## 🚀 Key Features

- **🧠 Multiclass SVM with RBF Kernel:** 98.50% overall accuracy, 97.70% macro F1-score, and sub-millisecond inference latency (0.061 ms/flow).
- **🛡️ 8-Console Streamlit SOC Dashboard:** Real-time KPI monitoring, threat analytics, manual flow triage, batch CSV triage, IP threat profiling, incident response ticketing, temporal evaluation, and Scapy packet capture.
- **⚡ FastAPI REST & WebSocket Backend:** Asynchronous `/ws/alerts` socket streaming live detections directly to monitoring consoles.
- **🔐 JWT Authentication & RBAC:** Secure PBKDF2/bcrypt password hashing with role-based access control (`admin`, `analyst`).
- **🌐 IP Reputation & Edge Firewall Toggles:** Automatic reputation scoring (0.0–100.0) tracking host behavior, with one-click edge IP blocking/unblocking.
- **📑 Incident Investigation Management:** Formal security incident tracking (`OPEN`, `IN_PROGRESS`, `RESOLVED`, `CLOSED`) with investigator audit timestamps.
- **📄 Executive PDF Audit Reports:** Instant ReportLab PDF generation compiling executive threat metrics, active high-severity alerts, and malicious IP profiles.
- **📅 Temporal Generalization (Day-Wise Cross-Testing):** Cross-validation across Monday–Friday CIC-IDS2017 attack profiles to verify zero temporal data leakage.
- **⚖️ Confidence Calibration & Anomaly Rejection:** Platt-scaled posterior probabilities with a 55.0% confidence cutoff isolating polymorphic zero-day attacks.
- **🗄️ Multi-Database Support:** Transparent zero-config SQLite for local development and PostgreSQL for high-concurrency production deployments.

---

## 🏗️ System Architecture

```
                                  [ Network Packet Ingestion ]
                                  (Scapy Live Capture / CIC-IDS2017)
                                                 │
                                                 ▼
                                     [ Flow Feature Extractor ]
                                    (71 Statistical Flow Metrics)
                                                 │
                                                 ▼
                                    [ Preprocessing Pipeline ]
                           (Median Imputer -> StandardScaler -> Leakage Filter)
                                                 │
                                                 ▼
                                [ Multiclass RBF Kernel SVM ]
                               (Class-Weighted Support Vectors)
                                                 │
                                                 ▼
                                  [ Confidence Calibration ]
                             ┌───────────────────┴───────────────────┐
                    Confidence ≥ 55%                        Confidence < 55%
                             │                                       │
                             ▼                                       ▼
                 [ Canonical Threat Label ]              [ UNKNOWN / ZERO-DAY ANOMALY ]
                             │                                       │
                             └───────────────────┬───────────────────┘
                                                 │
                                                 ▼
                                      [ Database Engine ]
                                  (SQLAlchemy: SQLite / PostgreSQL)
                                 - Predictions  - Security Alerts
                                 - IP Tracking  - Incident Records
                                                 │
                                                 ├──────────────────────────────────┐
                                                 ▼                                  ▼
                                      [ FastAPI Microservice ]            [ WebSocket Hub ]
                                         - REST Endpoints                  (/ws/alerts)
                                         - JWT Auth (RBAC)                          │
                                         - ReportLab PDF Engine                     │
                                                 │                                  │
                                                 └──────────────────┬───────────────┘
                                                                    ▼
                                                    [ Streamlit SOC Dashboard ]
                                                     (8 Specialized Consoles)
```

---

## 🛡️ SOC Operational Consoles

The Streamlit front-end provides 8 dedicated operations consoles:

| Console | Capabilities |
| :--- | :--- |
| **1. Security Overview** | Executive KPI cards (Total Flows, Detected Attacks, Normal Flows, Zero-Day Anomalies), interactive attack donut chart, and live security alerts with one-click triage buttons (`INVESTIGATING`, `RESOLVED`, `FALSE_POSITIVE`). |
| **2. Threat Analytics & Timeline** | Hourly threat velocity timeline, category distribution bar charts, severity breakdown (Critical, High, Medium, Low), and attack risk matrix. |
| **3. Intrusion Detection Engine** | Interactive manual flow parameter testing + High-throughput CSV batch prediction with automated audit database logging. |
| **4. IP Monitoring & Reputation** | Host profiling table, reputation scoring (0–100), threat levels (SAFE, SUSPICIOUS, MALICIOUS), and interactive **🚫 Block IP / ✓ Unblock IP** firewall toggles. |
| **5. Incident Investigation** | Formal incident ticket tracking (`INC-XXXX`), investigator notes, status updates (`OPEN`, `IN_PROGRESS`, `RESOLVED`, `CLOSED`), and new incident creation form. |
| **6. Model Performance & Metrics** | 3 comprehensive tabs: Multiclass Metrics & Confusion Matrix heatmap, **Day-Wise Temporal Generalization** table & multi-panel chart, and Confidence Calibration analysis. |
| **7. Executive Reports & Audit Logs** | One-click **📥 Download Executive PDF Audit Report** (ReportLab generated) and complete CSV historical audit trail export. |
| **8. Live Network Monitoring** | Scapy packet sniffer controls, active bidirectional flow reassembly buffer, interface selection, and real-time alert trigger. |

---

## 📊 Dataset & Canonical Attack Taxonomy

Trained on the **CIC-IDS2017** benchmark dataset provided by the Canadian Institute for Cybersecurity:

| Canonical Label | Original Attack Vectors Included | Severity |
| :--- | :--- | :---: |
| **BENIGN** | Normal Web browsing, DNS lookups, SSH management, background OS traffic | `LOW` |
| **DoS** | DoS Hulk, DoS GoldenEye, DoS Slowloris, DoS Slowhttptest | `HIGH` |
| **DDoS** | Distributed Denial of Service (LOIC HTTP / UDP Floods) | `CRITICAL` |
| **PortScan** | Nmap / Reconnaissance port sweeps and service probes | `MEDIUM` |
| **Bot** | Botnet Command & Control (C2) beaconing (Ares Botnet) | `HIGH` |
| **FTP-Patator** | High-velocity dictionary brute force against Port 21 (FTP) | `HIGH` |
| **SSH-Patator** | Distributed dictionary brute force against Port 22 (SSH) | `HIGH` |
| **Web Attack** | SQL Injection (SQLi), Cross-Site Scripting (XSS), Web Brute Force | `CRITICAL` |
| **Infiltration** | Post-exploitation lateral movement and internal exfiltration | `CRITICAL` |
| **Heartbleed** | OpenSSL TLS Heartbeat buffer overflow vulnerability | `CRITICAL` |

---

## 🧠 Machine Learning Engine

### Why Support Vector Machine (SVM) with RBF Kernel?
In network traffic analysis, attack patterns frequently exhibit non-linear boundaries across statistical flow dimensions (such as packet size variance and inter-arrival times). The **Radial Basis Function (RBF)** kernel projects vectors into an infinite-dimensional reproducing kernel Hilbert space:

$$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$

- **Hyperparameters:** Optimized via `GridSearchCV` with stratified 3-fold cross-validation ($C=10.0, \gamma=\text{'scale'}$, class-balanced sample weights).
- **Leakage Prevention:** Identifiers (`Flow ID`, `Source IP`, `Destination IP`, `Timestamp`) are explicitly excluded from model features, ensuring the classifier learns true flow dynamics.
- **Zero-Day Anomaly Rejection:** Posterior class probabilities are calibrated using Platt scaling. When maximum class probability falls below **55.0%**, traffic is flagged as `UNKNOWN / ANOMALY`, alerting analysts to novel threat patterns.

### Model Evaluation Results (1,200 Unseen Test Records)

| Metric | Measured Result |
| :--- | :--- |
| **Overall Accuracy** | **98.50%** |
| **Macro F1-Score** | **97.70%** |
| **Weighted F1-Score** | **98.55%** |
| **Prediction Latency** | **0.061 ms / flow** |

---

## 📅 Day-Wise Temporal Generalization Results

To prove that the model does not suffer from temporal data leakage and generalizes across varied attack types, the model was cross-tested across all 5 days of the CIC-IDS2017 schedule (**15,000 total flows**):

| Day | Scenario / Threat Vectors | Total Flows | Attack Flows | Accuracy | Attack Recall | Benign FPR | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Monday** | Baseline Normal Traffic (No Attacks) | 1,500 | 0 | **97.87%** | 100.0% | 2.13% | 0.167 ms |
| **Tuesday** | Brute Force (FTP-Patator, SSH-Patator) | 2,400 | 900 | **98.25%** | **99.89%** | 2.60% | 0.161 ms |
| **Wednesday** | Volumetric DoS & Heartbleed | 3,450 | 1,950 | **99.13%** | **99.95%** | 1.93% | 0.142 ms |
| **Thursday** | Web Attacks (SQLi, XSS) & Infiltration | 2,250 | 750 | **97.42%** | **98.67%** | 2.67% | 0.133 ms |
| **Friday** | DDoS, PortScan & Botnet | 5,400 | 3,900 | **99.19%** | **99.90%** | 2.53% | 0.123 ms |

*Day-wise charts saved to `reports/day_wise_evaluation.png` and `reports/day_wise_report.json`.*

---

## 🔌 API Specification & Real-Time WebSockets

### Core Endpoints

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :---: |
| `GET` | `/health` | System health, database connectivity, and loaded model verification | Public |
| `POST` | `/auth/login` | OAuth2 password flow returning signed JWT access token | Public |
| `GET` | `/auth/me` | Current user profile and assigned role (`admin` / `analyst`) | Bearer JWT |
| `POST` | `/predict` | Single flow classification with automatic alert generation | Public |
| `POST` | `/predict/batch` | High-throughput batch CSV flow classification | Public |
| `GET` | `/analytics/overview`| Summary metrics (total flows, attack count, top attack) | Public |
| `GET` | `/soc/incidents` | List tracked security incidents (with status filters) | Public |
| `POST` | `/soc/incidents` | Create formal cybersecurity incident ticket | Public |
| `PATCH`| `/soc/incidents/{code}`| Update incident status and add investigator notes | Public |
| `GET` | `/soc/ips` | Retrieve profiled IP reputations and threat levels | Public |
| `POST` | `/soc/ips/block` | Toggle firewall block/unblock rule on target IP | Public |
| `PATCH`| `/soc/alerts/{id}/status`| Update alert triage status | Public |
| `GET` | `/soc/day-wise-evaluation`| Retrieve temporal generalization benchmark data | Public |
| `GET` | `/soc/reports/pdf` | Download ReportLab executive security PDF report | Public |
| `WS` | `/ws/alerts` | Real-time WebSocket connection broadcasting threat alerts | Public |

---

## 🗄️ Multi-Database Architecture

The data tier is abstracted using **SQLAlchemy ORM**:
- **SQLite:** Defaults to `sqlite:///logs/predictions.db` for zero-configuration local development.
- **PostgreSQL:** Activated automatically whenever the `DATABASE_URL` environment variable is defined:
  ```bash
  export DATABASE_URL="postgresql://user:password@hostname:5432/nids_db"
  ```
- **Auto-Migration:** Schema columns and foreign keys are verified at startup via `database.init_db()`.
- **Default Seeding:** `admin` and `analyst` accounts are provisioned on initial boot.

---

## 💻 Installation & Execution Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.12)
- Windows / Linux / macOS
- 8 GB RAM

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/THARUN-RK/Network-intrusion-detection-system-ML.git
cd Network-intrusion-detection-system-ML

# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Or on Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Services

**Option A — Unified Interactive Launcher:**
```bash
python app.py
```

**Option B — Independent Services:**
```bash
# Terminal 1: Launch FastAPI REST & WebSocket Backend
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Launch Streamlit SOC Dashboard
streamlit run frontend/dashboard.py --server.port 8501
```

Access the dashboard at `http://localhost:8501` and API docs at `http://localhost:8000/docs`.

---

## 🐳 Docker Container Deployment

Run the complete multi-container stack with PostgreSQL, FastAPI, and Streamlit:

```bash
docker-compose up --build -d
```

- **Streamlit SOC Dashboard:** `http://localhost:8501`
- **FastAPI API & Docs:** `http://localhost:8000/docs`
- **PostgreSQL Database:** `localhost:5432` (`nids_db`)

---

## 🧪 Automated Testing Suite

The repository contains 20 comprehensive unit and integration tests covering data preprocessing, model inference, live flow generation, REST endpoints, incident workflows, and day-wise temporal evaluation:

```bash
pytest -v
```

```text
============================= test session starts =============================
collected 20 items

tests/test_api.py::test_health_check_function PASSED                     [  5%]
tests/test_api.py::test_model_info_function PASSED                       [ 10%]
tests/test_api.py::test_predict_single_endpoint PASSED                   [ 15%]
tests/test_api.py::test_predict_batch_endpoint PASSED                    [ 20%]
tests/test_api.py::test_analytics_and_history_endpoints PASSED           [ 25%]
tests/test_api.py::test_auth_login PASSED                                [ 30%]
tests/test_api.py::test_soc_incident_workflow PASSED                     [ 35%]
tests/test_api.py::test_soc_ip_reputation_and_block PASSED               [ 40%]
tests/test_api.py::test_soc_pdf_report_generation PASSED                 [ 45%]
tests/test_evaluation.py::test_partition_dataset_by_days PASSED          [ 50%]
tests/test_evaluation.py::test_day_wise_evaluation_execution PASSED      [ 55%]
tests/test_monitoring.py::test_network_flow_feature_extraction PASSED    [ 60%]
tests/test_monitoring.py::test_flow_features_compatibility_with_model PASSED [ 65%]
tests/test_prediction.py::test_predictor_loaded PASSED                   [ 70%]
tests/test_prediction.py::test_predict_single_flow PASSED                [ 75%]
tests/test_prediction.py::test_predict_batch PASSED                      [ 80%]
tests/test_preprocessing.py::test_clean_column_name PASSED               [ 85%]
tests/test_preprocessing.py::test_canonicalize_label PASSED              [ 90%]
tests/test_preprocessing.py::test_network_flow_sanitizer PASSED          [ 95%]
tests/test_preprocessing.py::test_feature_pipeline_fit_transform PASSED  [100%]

============================= 20 passed in 16.73s =============================
```

---

## 🎓 Hardware Optimization & CSE Presentation Guide

### Why Not Deep Learning (e.g. CNN / LSTM)?
> *"Deep neural networks require GPU acceleration, millions of parameters, and introduce significant latency overhead (>5 ms/flow). Our RBF SVM achieves 98.50% accuracy with an inference latency of just 0.061 ms/flow, making it optimal for line-rate edge network intrusion detection."*

### How Was Class Imbalance Handled?
> *"Rare attacks like Heartbleed and Infiltration represent <1% of the raw data. We handled this using Scikit-Learn's `class_weight='balanced'` in our SVM loss formulation, which scales penalties inversely proportional to class frequencies."*

### Why Day-Wise Testing Matters:
> *"In the CIC-IDS2017 dataset, random train-test splitting can result in temporal data leakage if consecutive flows share TCP connection states. By performing day-wise evaluation across Monday through Friday, we demonstrated that detection rates remain >98% across completely distinct attack distributions."*

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
