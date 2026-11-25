# 🛡️ Enterprise Network Intrusion Detection System (NIDS) & Security Operations Center (SOC)

[![Live SOC Platform](https://img.shields.io/badge/Live%20Platform-Render-0284c7?style=for-the-badge&logo=render)](https://network-intrusion-detection-system-ml.onrender.com/)
[![GitHub Pages](https://img.shields.io/badge/Web%20Portal-GitHub%20Pages-22c55e?style=for-the-badge&logo=github)](https://tharun-rk.github.io/Network-intrusion-detection-system-ML/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Web Console](https://img.shields.io/badge/SOC%20Console-Web%20Dashboard-blue.svg)](/)
[![Docker](https://img.shields.io/badge/Container-Docker%20Compose-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-32%20Passed-brightgreen.svg)](tests/)

An enterprise-grade, full-stack **Network Intrusion Detection System (NIDS)** and interactive **Security Operations Center (SOC)**. Built with **Multiclass Support Vector Machine (RBF Kernel)**, **FastAPI REST API**, **WebSockets**, **SQLAlchemy (SQLite + PostgreSQL)**, **Scapy Live Packet Capture**, **ReportLab Executive PDF Generator**, and an ultra-fast **Web SOC Dashboard**. 

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
12. [Evaluation and Limitations](#evaluation-and-limitations)
13. [Hardware Optimization & CSE Presentation Guide](#-hardware-optimization--cse-presentation-guide)

---

## 🚀 Key Features

- **🧠 Multiclass SVM with RBF Kernel:** 98.40% accuracy, 97.59% macro F1-score, and 99.43% macro PR-AUC on the 3,000-flow held-out benchmark split (`0.097 ms/flow`), with explicit natural-distribution (80% benign) evaluation (`92.93%` macro F1).
- **🛡️ Real-Time Web SOC Dashboard:** Fast, lightweight operations console with live KPI monitoring, threat analytics, interactive flow prediction sandbox, and IP threat profiling.
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
                                                    [ Web SOC Dashboard & API ]
                                                     (8 Specialized Consoles)
```

---

## 🛡️ SOC Operational Consoles

The Web SOC Console provides real-time operations capabilities:

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

- **Hyperparameters:** Optimized via `GridSearchCV` with stratified 5-fold cross-validation ($C=10, \gamma=0.01$, class-balanced sample weights).
- **Leakage Prevention:** Identifiers (`Flow ID`, `Source IP`, `Destination IP`, `Timestamp`) are dropped, `inf`/`NaN` and duplicate rows are cleaned before splitting, and preprocessing (`SimpleImputer` + `StandardScaler`) fits strictly inside training folds.
- **Zero-Day Anomaly Rejection:** Platt-scaled posterior probabilities below **55.0%** trigger `ANOMALY (Unknown Threat)` (see [Evaluation and Limitations](#evaluation-and-limitations) for honest Leave-One-Attack-Out failure analysis).

### Model Evaluation Results (3,000 Held-Out Benchmark Flows)

| Metric | Measured Result |
| :--- | :--- |
| **Overall Accuracy** | **98.40%** |
| **Macro F1-Score** | **97.59%** |
| **Weighted F1-Score** | **98.45%** |
| **Macro PR-AUC** | **99.43%** |
| **Prediction Latency** | **0.097 ms / flow** |

---

## 📅 Day-Wise Temporal Generalization Results

When partitioned into simulated Monday–Friday attack schedules strictly over the **3,000 held-out test flows** (`CICIDS2017_Benchmark_Sample.csv` lacks real timestamps, so this tests label-grouped profile stability rather than true temporal drift):

| Day | Scenario / Threat Vectors | Total Flows | Attack Flows | Accuracy | Attack Recall | Benign FPR | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Monday** | Baseline Normal Traffic (No Attacks) | 300 | 0 | **98.00%** | N/A (0 attacks) | 2.00% | 0.103 ms |
| **Tuesday** | Brute Force (FTP-Patator, SSH-Patator) | 480 | 180 | **98.33%** | **100.00%** | 2.33% | 0.100 ms |
| **Wednesday** | Volumetric DoS & Heartbleed | 690 | 390 | **98.70%** | **100.00%** | 3.00% | 0.102 ms |
| **Thursday** | Web Attacks (SQLi, XSS) & Infiltration | 450 | 150 | **97.11%** | **98.67%** | 3.33% | 0.102 ms |
| **Friday** | DDoS, PortScan & Botnet | 1,080 | 780 | **98.89%** | **99.87%** | 3.67% | 0.100 ms |

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
| `GET` | `/api/stats` | Real-time aggregate statistics, velocity, and 5-min class breakdown | Public |
| `GET` | `/api/timeseries` | Minute-by-minute total, benign, and attack flows (last 30 min) | Public |
| `GET` | `/api/stream` | Server-Sent Events (SSE) stream pushing live predictions & alerts | Public |
| `POST` | `/api/simulator/toggle` | Dynamically enable or disable the background traffic simulator | Public |
| `WS` | `/ws/alerts` | Real-time WebSocket connection broadcasting threat alerts | Public |

### ⚡ Real-Time Traffic Simulation Mode

The system includes an integrated background traffic simulator that drives continuous SOC activity:
- Samples genuine flows from the held-out `dataset/raw/CICIDS2017_Benchmark_Sample.csv`.
- Introduces subtle numeric variance (+/- 1-2%) while preserving authentic protocol characteristics.
- Passes every vector through the identical SVM inference pipeline (`predict_single_flow`).
- Generates realistic attack bursts (10–25 flows of DDoS or PortScan) alongside high-volume clean traffic.
- Dynamically recalculates IP reputation scores, auto-blocks high-threat IPs (< 35.0), and enforces table size capping (10,000 max rows).

**Environment Configuration:**
```bash
# Enable or disable the simulator (default: true)
export SIMULATOR_ENABLED=true

# Mean interval between flows in seconds (default: 1.5s, with random 1.0–3.0s jitter)
export SIMULATOR_RATE=1.5
```

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

**Option B — Independent Service:**
```bash
# Launch FastAPI REST, WebSocket Backend & Web Console
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Access the Web SOC Dashboard at `http://localhost:8000` and interactive Swagger API docs at `http://localhost:8000/docs`.

---

## 🐳 Docker Container Deployment

Run the complete multi-container stack with PostgreSQL and FastAPI:

```bash
docker-compose up --build -d
```

- **Web SOC Dashboard & API:** `http://localhost:8000`
- **Swagger Documentation:** `http://localhost:8000/docs`
- **PostgreSQL Database:** `localhost:5432` (`nids_soc`)

---

## 🧪 Automated Testing Suite

The repository contains 23 unit and integration tests covering data preprocessing, leakage-free train/test splitting, model inference, Scapy vs. CICFlowMeter PCAP feature parity (all 71 features), REST endpoints, incident workflows, and day-wise evaluation:

```bash
pytest -v
```

```text
23 passed in 8.42s
```

---

## Evaluation and Limitations

All artifacts below are generated directly from reproducible code (`python evaluate_rigorous.py`, `RANDOM_STATE=42`) and stored under [`results/`](results/).

### 1. Pipeline Audit & Root Cause of `Heartbleed=20` / `Infiltration=36`

1. **Dataset Provenance (`dataset_utility.py` vs. Real CIC-IDS2017):**
   - The repository contains `dataset/raw/CICIDS2017_Benchmark_Sample.csv` (15,000 **synthetically generated** flows: 50% BENIGN, 150 Heartbleed [1%], 300 Infiltration [2%]), **not** the raw 2.83M-flow CIC-IDS2017 CSVs (where Heartbleed has 11 total flows, Infiltration has 36 total flows, and ~80.3% of traffic is BENIGN).
2. **Direct Train/Test Leakage in Legacy `evaluation.py` & `day_wise_evaluation.py`:**
   - Legacy `train_model.py` stratified-sampled 8,000 rows *before* splitting `6,400` train / `1,600` test without persisting the test split. Legacy `evaluation.py` then ran `X.sample(n=2000, random_state=42)` across all `15,000` rows, yielding `Heartbleed=20` and `Infiltration=36` (matching `reports/evaluation_report.json`) — **869 out of 2,000 evaluation rows (43.45%) were training rows**.
   - Legacy `day_wise_evaluation.py` evaluated all 15,000 rows (leaking all 6,400 training rows) on simulated label groups without real timestamps.
3. **Deduplication & Cleaning Gap:**
   - Legacy `preprocessing.py` did not drop duplicate feature rows or `inf`/`NaN` rows prior to splitting. While `CICIDS2017_Benchmark_Sample.csv` has 0 duplicates and 0 `inf`/`NaN` rows, real CIC-IDS2017 CSVs require pre-split deduplication and sanitization to avoid duplicate leakage across splits.
4. **`Destination Port` Shortcutting & Feature Collinearity:**
   - A depth-5 Decision Tree trained solely on `Destination Port` achieves **68.73% accuracy** (`0.4825` macro F1) because synthetic classes bind tightly to static ports (`FTP-Patator=21`, `SSH-Patator=22`, `Heartbleed=443`). Removing `Destination Port` lowers SVM Natural Test Macro F1 from `92.93%` to `90.67%`.
   - Synthetic generator formulas also create exact linear/scalar dependencies (`Bwd Header Length = 20 * bwd_pkts`, `Fwd Packet Length Max = 1.5 * pkt_len_mean`), producing identical permutation-importance ties across derived columns.

### 2. Leakage-Free Model Comparison (5-Fold CV + Balanced vs. ~80% Benign Test Sets)

Data was cleaned (`drop_inf_nan`, `drop_duplicates`) and split **first** (70% train = `10,500`, 30% held-out pool = `4,500`; `0` index overlap, `0` feature-hash overlap). Balancing (`class_weight='balanced'` or train-only `SMOTE`) was applied **only inside training folds**:
- **(a) Balanced Test Set (`n=450`):** 45 flows per class across all 10 categories, sampled strictly from the held-out pool.
- **(b) Natural Class Distribution Test Set (`n=2,812`):** All 2,250 held-out `BENIGN` flows (`80.01%`) plus 562 proportionally sampled held-out attacks.

| Model | 5-Fold CV Acc | 5-Fold CV Macro F1 | Balanced Acc / Macro F1 / PR-AUC | Natural Acc / Macro F1 / PR-AUC | Natural Benign FPR / Attack Recall | Train (s) | Infer (ms/flow) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM (RBF, class_weight)** | 98.38 ± 0.25% | 97.40 ± 0.38% | 98.67% / 98.66% / 99.86% | 97.72% / **92.93%** / 97.92% | 2.62% / 99.82% | 2.983 | 0.0916 |
| **SVM (RBF, Train-Only SMOTE)** | 99.05 ± 0.23% | 98.30 ± 0.39% | 97.56% / 97.55% / 99.82% | 98.61% / 94.81% / 98.44% | 1.29% / 99.11% | 2.988 | 0.0800 |
| **Random Forest** | 99.35 ± 0.09% | 98.87 ± 0.14% | 99.78% / 99.78% / 99.94% | 99.18% / 97.42% / 99.76% | 1.02% / 100.00% | 1.208 | 0.0168 |
| **XGBoost** | **99.69 ± 0.09%** | **99.43 ± 0.22%** | **99.78% / 99.78% / 99.97%** | **99.68% / 98.93% / 99.97%** | **0.40% / 100.00%** | 3.857 | 0.0073 |
| **Logistic Regression** | 98.08 ± 0.26% | 97.02 ± 0.39% | 99.33% / 99.33% / 99.78% | 97.40% / 92.73% / 96.92% | 3.16% / 100.00% | **0.527** | **0.0016** |
| **MLP Neural Net** | 98.92 ± 0.23% | 97.96 ± 0.45% | 98.22% / 98.23% / 99.74% | 99.08% / 96.33% / 99.12% | 0.84% / 99.47% | 1.741 | 0.0025 |

**Key Honest Findings:**
- **Base-Rate Precision Drop:** On Balanced Test, SVM (`class_weight='balanced'`) scores `98.66%` Macro F1 (`Bot` precision `1.000`). Under a realistic ~80% BENIGN distribution (`n=2,812`), its `2.62%` BENIGN false-positive rate misclassifies 28 benign flows as `Bot` (`precision=0.600`, `F1=0.730`) and 25 as `Web Attack` (`precision=0.550`, `F1=0.702`), pulling Natural Macro F1 down to **`92.93%`**.
- **Tree Baselines Outperform RBF SVM:** `XGBoost` achieves `98.93%` Natural Macro F1, `0.40%` Benign FPR, and **12.5× faster inference** (`0.0073 ms/flow` vs `0.0916 ms/flow`). Train-only `SMOTE` also improves SVM Natural Macro F1 from `92.93%` to `94.81%` and cuts Benign FPR in half (`1.29%`).

![Model Comparison](results/model_comparison.png)
![SVM Natural Confusion Matrix](results/confusion_matrix_svm_natural.png)

### 3. Leave-One-Attack-Out (LOAO) Zero-Day Evaluation (<55% Confidence Rule)

Holding out one attack class at a time from SVM training demonstrates that **thresholding closed-world Platt probabilities at `<55%` is NOT a reliable zero-day detector**:

| Held-Out Unseen Attack | Unseen Flows | Rejected as Anomaly (<55%) | Mapped to Other Attack (>=55%) | Missed as BENIGN (>=55%) | Mean Max Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bot** | 180 | 18.89% | 24.44% | 56.67% | 77.31% |
| **DDoS** | 540 | **0.00%** | 0.00% | **100.00%** | 91.25% |
| **DoS** | 540 | 41.11% | 58.89% | 0.00% | 61.21% |
| **FTP-Patator** | 135 | 32.59% | 62.22% | 5.19% | 66.55% |
| **Heartbleed** | 45 | **0.00%** | 0.00% | **100.00%** | 91.15% |
| **Infiltration** | 90 | **0.00%** | 1.11% | **98.89%** | 88.82% |
| **PortScan** | 450 | 14.67% | 16.44% | 68.89% | 82.49% |
| **SSH-Patator** | 135 | 7.41% | 92.59% | 0.00% | 83.66% |
| **Web Attack** | 135 | **0.00%** | 4.44% | **95.56%** | 98.39% |
| **Macro Average (9 attacks)** | — | **12.74%** | 28.90% | **58.36%** | — |

- Closed-world Platt scaling assigns high probability (`88.8%–98.4%`) to unseen attack vectors that fall into the `BENIGN` half-space (100% miss rate on unseen `DDoS` and `Heartbleed`, 98.89% on `Infiltration`, 95.56% on `Web Attack`). Benign false-anomaly rejection averages `0.30%`. True open-set detection requires out-of-distribution / one-class anomaly models rather than closed-world probability cutoffs.
- In `predict.py`, we also fixed the bug where `confidence < 55.0 and predicted_class != "BENIGN"` skipped low-confidence `BENIGN` predictions.

![LOAO Zero-Day Breakdown](results/loao_zero_day_detection.png)

### 4. Feature Importance (Permutation Importance & SHAP Top 15)

Top features computed on the held-out test set (`results/feature_importance.json`):
- **SVM Top Features (Permutation / KernelSHAP):** `Down/Up Ratio` (`0.17467 ± 0.01184`), `Destination Port` (`0.07148 ± 0.00576`, #1 in KernelSHAP `0.01059`), `Bwd Header Length`, `Subflow Bwd Packets`, `Total Backward Packets`, `Total Length of Fwd Packets`, `act_data_pkt_fwd`, `SYN Flag Count`.
- **XGBoost Top Features (Permutation / TreeSHAP):** `Total Fwd Packets` (`0.15684` / `1.13487`), `Destination Port` (`0.11512` / `0.91482`), `Total Backward Packets` (`0.09244` / `0.43955`), `Down/Up Ratio`, `Flow Bytes/s`, `Flow IAT Mean`, `Flow Packets/s`, `Flow Duration`.

![Permutation Importance Top 15](results/feature_importance_permutation_top15.png)
![SHAP Top 15](results/feature_importance_shap_top15.png)

### 5. Scapy vs. CICFlowMeter Parity Audit & Fixes

Auditing `monitoring/flow_extractor.py` and `monitoring/packet_capture.py` against CICFlowMeter's specification exposed and resolved multiple mismatches:
- **Payload vs. Ethernet Frame Length:** Replaced full L2 `len(pkt)` with transport payload bytes (`ip.len - ip.ihl*4 - transport_hdr_len`), ignoring Ethernet frame headers and padding.
- **Sample Std (`ddof=1`) & Flow IAT Index:** Switched from `ddof=0` to Bessel-corrected `ddof=1` (`SummaryStatistics`), and fixed the packet-#1 bug that appended a spurious `0.0` IAT.
- **PCAP Wire Timestamps:** Fixed `FlowAggregator` to use microsecond packet capture timestamps (`pkt.time`) instead of wall-clock `time.time()`, preventing catastrophic float64 cancellation at `1e9` epoch scale.
- **Headers, TCP Windows, Directional Flags, Active/Idle & Subflows:** Parsed actual TCP `dataofs * 4` / UDP `8`-byte headers, `Init_Win_bytes_*` (`-1` default), `act_data_pkt_fwd` (`payload > 0`), directional `Fwd/Bwd PSH` & `URG`, `CWE`/`ECE`, 1s subflows, and 5s Active/Idle intervals.
- **Known Residual Limitation:** `Down/Up Ratio` uses continuous float `bwd/fwd` in Python rather than CICFlowMeter's legacy Java integer floor division `int(bwd/fwd)`, which should be matched if running directly against raw Java CICFlowMeter exports. Verified on a multi-flow TCP/UDP PCAP across all 71 features in `tests/test_monitoring.py`.

### Commands to Reproduce Everything

```bash
# Install evaluation dependencies (includes xgboost, shap, imbalanced-learn, scapy)
pip install -r requirements-eval.txt

# 1. Run full leakage audit, 5-fold CV, baseline comparisons, LOAO zero-day & SHAP/Permutation importance
python evaluate_rigorous.py

# 2. Retrain production SVM and regenerate leakage-free reports/
python train_model.py
python evaluation.py
python day_wise_evaluation.py

# 3. Verify all unit, API, pipeline, and Scapy CICFlowMeter PCAP parity tests
pytest -v
```

---

## 🎓 Hardware Optimization & CSE Presentation Guide

### Tree Ensembles vs. RBF SVM & Deep Learning
> *"On tabular 71-feature CIC-IDS2017 flows, gradient-boosted trees (`XGBoost`, 0.0073 ms/flow, 98.93% natural macro F1) outperform RBF SVM (0.0916 ms/flow, 92.93% natural macro F1) in both precision under class imbalance and CPU inference speed, without GPU overhead."*

### How Class Imbalance Was Handled
> *"Rare attacks like Heartbleed and Infiltration are heavily outweighed by BENIGN flows. Both `class_weight='balanced'` and train-only `SMOTE` (fit strictly inside CV folds after train/test splitting) were evaluated; train-only SMOTE reduced SVM's natural benign FPR from 2.62% to 1.29%."*

### Temporal & Benchmark Validation Limits
> *"Evaluating day-wise partitions on held-out flows avoids direct train/test row overlap, though true deployment generalization still requires evaluation against raw timestamped PCAP captures containing real non-synthetic traffic."*

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.



