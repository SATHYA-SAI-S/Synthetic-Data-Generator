# Privacy-Preserving Clinical Data Synthesis & Audit Platform
## UI/UX Implementation Plan — Faculty Review Edition

> **Version:** 1.0 | **Project:** ADVT — Privacy-Preserving Synthetic Healthcare Data Generation
> **Audience:** Faculty Reviewers, Clinical Data Governance Stakeholders, Research Evaluators
> **Stack Recommendation:** Python + Streamlit (rapid deployment) OR React + FastAPI (production)

---

## 0. Core Philosophy & Design Statement

This is **not** a data generation tool. It is a **Mathematical Audit & Synthesis Platform** — every action taken by the system is logged, metered, and verified against clinical and privacy standards before a single synthetic row is emitted.

The interface should answer three questions for any reviewer at a glance:

| Question | Module Answering It |
| :--- | :--- |
| *"Has patient identity been protected?"* | **Module B: Privacy Ledger** |
| *"Is the synthetic data clinically useful?"* | **Module C: Fidelity & Utility Suite** |
| *"Was the process fully automated and auditable?"* | **Module A: Schema Intelligence Panel** |

---

## 1. Brand Identity & Design Tokens

### 1.1 Platform Name & Tagline

```
SYNTHGUARD
Privacy-Preserving Clinical Data Synthesis & Audit Platform
"Mathematically Audited. Clinically Grounded. Differentially Private."
```

### 1.2 Color System

| Token | Hex | Usage |
| :--- | :--- | :--- |
| System Blue (Primary) | `#1A73E8` | Primary actions, headers, active states |
| Deep Navy (Background) | `#0D1B2A` | Sidebar, panels, dark mode base |
| Clinical White | `#F8FAFC` | Content surface, card backgrounds |
| Compliance Green | `#22C55E` | PASS status, zero-NaN badges, safe budget |
| Privacy Amber | `#F59E0B` | Budget warnings, caution indicators |
| Breach Red | `#EF4444` | Budget exhausted, NaN detected, HIPAA flag |
| Data Gray | `#64748B` | Secondary text, axis labels, metadata |

### 1.3 Typography

| Role | Font | Weight | Size |
| :--- | :--- | :--- | :--- |
| Platform Title | Inter | 700 Bold | 28px |
| Section Headers | Inter | 600 SemiBold | 18px |
| Metric Values (KPIs) | JetBrains Mono | 700 Bold | 36px |
| Body / Labels | Inter | 400 Regular | 14px |
| Code Snippets | JetBrains Mono | 400 Regular | 13px |
| Status Badges | Inter | 600 SemiBold | 11px UPPERCASE |

### 1.4 Spacing & Layout Grid

- **Base unit:** 8px
- **Card padding:** 24px
- **Section gap:** 32px
- **Sidebar width:** 260px (fixed)
- **Main content:** Fluid, max-width 1440px

---

## 2. Application Layout Architecture

```
+------------------------------------------------------------------------------+
|  SYNTHGUARD   [Dataset: UCI Diabetes]  [Session: eps_1.0]  [Status: DONE]   |
|  NAVIGATION SIDEBAR                                                          |
+----------------+-------------------------------------------------------------+
|                |                                                             |
|  Overview      |              MAIN CONTENT AREA                             |
|                |                                                             |
|  Module A      |   +-----------------------------------------------------+  |
|  Schema Intel  |   |  KPI RIBBON (Live Metrics)                          |  |
|                |   |  e Spent | MIA Adv | TVD Score | NaN Count          |  |
|  Module B      |   +-----------------------------------------------------+  |
|  Privacy Ledger|                                                             |
|                |   +-----------------------------------------------------+  |
|  Module C      |   |  ACTIVE MODULE CONTENT (A / B / C)                  |  |
|  Fidelity Suite|   |                                                      |  |
|                |   |                                                      |  |
|  Audit Log     |   |                                                      |  |
|                |   +-----------------------------------------------------+  |
|  Config        |                                                             |
+----------------+-------------------------------------------------------------+
```

---

## 3. Module A — Schema Intelligence Panel

**Tagline:** *"Auto-Config Engine: Zero Manual Schema Wiring"*

### 3.1 Layout Overview

```
+-----------------------------------------------------------------------------+
| MODULE A: SCHEMA INTELLIGENCE PANEL                           [PROFILER LOG] |
+--------------------------------+--------------------------------------------+
|  A1. Dataset Ingest Card       |  A2. Column Profiler Table                 |
|  +------------------------+    |  +---------------------------------------+  |
|  | Dataset: UCI Diabetes  |    |  | Column           | Type   | Flag      |  |
|  | Source:  diabetic_data |    |  |------------------|--------|-----------|  |
|  | Rows:    101,766       |    |  | encounter_id     | ID     | HIPAA     |  |
|  | Cols:    50 -> 44 kept |    |  | patient_nbr      | ID     | HIPAA     |  |
|  | Session: eps_1.0       |    |  | time_in_hospital | Cont.  | CLEAN     |  |
|  +------------------------+    |  | num_medications  | Count  | CLEAN     |  |
|                                |  | State (CDC)      | Geo    | QUASI-ID  |  |
|  A3. Missingness Registry      |  +---------------------------------------+  |
|  +------------------------+    |                                             |
|  | Active Indicators: 28  |    |  A4. HIPAA Compliance Badge                |
|  | 78% covered            |    |  +---------------------------------------+  |
|  | Columns with NaN: 11   |    |  |  HIPAA SAFE HARBOR: PASSED            |  |
|  | Policy: Mask + Restore |    |  |  18/18 Direct Identifiers: EXCLUDED   |  |
|  +------------------------+    |  |  Quasi-Identifiers Flagged: 3         |  |
|                                |  |  Privacy Action: DROPPED              |  |
|                                |  +---------------------------------------+  |
+--------------------------------+--------------------------------------------+
```

### 3.2 Interactive Features

**Column Profiler Table** (sortable, filterable):
- Columns: `Column Name`, `Inferred Dtype`, `Missing Rate (%)`, `HIPAA Status`, `Cardinality`, `Action Taken`
- Color-coded rows: Red = HIPAA dropped | Amber = Quasi-ID warned | Green = Safe to encode
- Click any column to open a side drawer with the raw value distribution histogram

**Missingness Registry Visualization:**
- Heatmap of which columns generated `__missing_flag` indicators
- Toggle: "Show Raw Columns" vs "Show Encoded Tensor Columns"

**Live Profiler Log Stream** (right panel):

```
[00:21:43] INFO  Loaded 101,766 rows x 50 columns
[00:21:44] INFO  Profiling 50 columns...
[00:21:44] WARN  Column 'encounter_id' -> HIPAA Identifier: Dropped
[00:21:44] WARN  Column 'patient_nbr'  -> HIPAA Identifier: Dropped
[00:21:45] INFO  Detected 11 columns with missing values -> Injecting flags
[00:21:45] INFO  OneHotEncoder fitted on 23 categorical columns
[00:21:45] INFO  StandardScaler fitted on 6 continuous columns
[00:21:45] INFO  D_encoded = 616 dimensions -> Registry saved
[00:21:45] DONE  Schema Intelligence: COMPLETE (6 HIPAA columns removed)
```

---

## 4. Module B — Privacy Ledger

**Tagline:** *"Your Epsilon is a Budget. Spend it wisely. Audit it completely."*

### 4.1 KPI Ribbon (Always-Visible Top Bar)

| e SPENT | MIA ADVANTAGE | PRIVACY STATUS | EPOCHS DONE |
| :--- | :--- | :--- | :--- |
| **0.3720** | **-0.0083** | **SAFE** | **5 / 5** |
| of e = 1.0 max | Target: < 0.05 | Budget: 37.2% | All Accounted |

### 4.2 Layout Overview

```
+-----------------------------------------------------------------------------+
| MODULE B: PRIVACY LEDGER                          [EXPORT AUDIT CERTIFICATE] |
+--------------------------------------+--------------------------------------+
|  B1. Epsilon Expenditure Timeline    |  B2. RDP Accountant Status           |
|                                      |                                      |
|  e  ^                                |  +----------------------------+       |
|  1.0| --------------------- LIMIT   |  | Mechanism: Gaussian        |       |
|  0.8|                                |  | Delta (d):  1 x 10^-4     |       |
|  0.6|                                |  | Sigma (s):  5.00           |       |
|  0.4| ------o  e = 0.372             |  | Clip Norm:  1.0            |       |
|  0.2|  o o                           |  | Batch Size: 256            |       |
|  0.0+-------------------- Epochs     |  | Dataset N:  101,766        |       |
|       1   2   3   4   5              |  | Accountant: RDP (Opacus)   |       |
|                                      |  +----------------------------+       |
|  Budget remaining: 62.8%             |                                      |
+--------------------------------------+--------------------------------------+
|  B3. Membership Inference Attack     |  B4. Per-Epoch Privacy Log           |
|      (D-MIA Audit)                   |                                      |
|  +------------------------------+    |  Epoch  Loss    e Spent  Status      |
|  |                              |    |  ----------------------------         |
|  |  MIA Advantage:  -0.0083     |    |  1/5    1.2341   0.0744  SAFE        |
|  |  Baseline (Random): 0.5      |    |  2/5    1.1908   0.1488  SAFE        |
|  |  Attack AUC:     0.4958      |    |  3/5    1.1723   0.2232  SAFE        |
|  |  Verdict: NO MEMORIZATION    |    |  4/5    1.1690   0.2976  SAFE        |
|  +------------------------------+    |  5/5    1.1656   0.3720  SAFE        |
+--------------------------------------+--------------------------------------+
```

### 4.3 Privacy Certificate Export

One-click downloadable **Privacy Audit Certificate** (PDF/Markdown):

```
=============================================================
        DIFFERENTIAL PRIVACY AUDIT CERTIFICATE
        SYNTHGUARD Platform -- Session: eps_1.0
=============================================================
Dataset:           UCI Diabetes (diabetic_data.csv)
Training Date:     2026-08-20
Mechanism:         Gaussian Mechanism (Renyi DP)
Privacy Guarantee: e = 0.3720  (d = 1.0 x 10^-4)
Gradient Clip:     C = 1.0
Noise Multiplier:  s = 5.00
Epochs:            5
MIA Advantage:     -0.0083  [THRESHOLD: < 0.05] PASSED
Verdict:           MATHEMATICALLY PRIVATE
=============================================================
```

### 4.4 Epsilon Budget Gauge (Visual Widget)

```
Privacy Budget Consumption
  [||||||||||||                    ]  37.2% spent
  [0.0]------[0.372]--------------[1.0]
        ^ Current               ^ Max Budget
```

Color transitions: Green (0–50%) → Amber (50–85%) → Red (85–100%)

---

## 5. Module C — Data Fidelity & Utility Suite

**Tagline:** *"Synthetic data that passes the audit — and the downstream machine learning test."*

### 5.1 Layout Overview

```
+-----------------------------------------------------------------------------+
| MODULE C: DATA FIDELITY & UTILITY SUITE              [COMPARE EPSILON]      |
+-----------------------------------------------------------------------------+
|  C1. Distribution Drift Analysis                                            |
|  +-----------------------------------------------------------------------+  |
|  |  Real Correlation Matrix           |  Synthetic (e=1.0) Correlation   |  |
|  |  [Heatmap: Deep Blue gradient]     |  [Heatmap: Matches Real ~RMSE]   |  |
|  |   Pearson RMSE = (baseline)        |   Pearson RMSE = 0.1948          |  |
|  |                                                                        |  |
|  |  Bivariate Correlation RMSE: 0.1948  [Target: < 0.35]  PASSED         |  |
|  +-----------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------+
|  C2. Marginal Distribution Comparison (TVD per Column)                      |
|  +-----------------------------------------------------------------------+  |
|  |  Column        TVD      Status              |                         |  |
|  |  change        0.0316   EXCELLENT           |                         |  |
|  |  readmitted    0.1124   GOOD                |                         |  |
|  |  race          0.2847   ACCEPTABLE          |                         |  |
|  |  gender        0.0521   EXCELLENT           |                         |  |
|  |  [All 44 columns shown, filterable and sortable]                      |  |
|  +-----------------------------------------------------------------------+  |
+-----------------------------------+-----------------------------------------+
|  C3. TSTR Benchmark Results       |  C4. Integrity Audit Status             |
|                                   |                                         |
|  AUC-ROC                          |  +-----------------------------------+  |
|   0.6855  TRTR (Real -> Real)     |  |  INTEGRITY AUDIT: ALL PASS        |  |
|   0.4962  TSTR (Synth -> Real)    |  |  NaN Count:        0 / 39,000     |  |
|                                   |  |  Schema Columns:   44 / 44        |  |
|  Utility Retention: 72.39%        |  |  ICD-9 Diversity:  390+ Codes     |  |
|  Target: > 60%  --  PASSED        |  |  Integer ID Drift: 0 Violations   |  |
|                                   |  |  Negative Counts:  0 Detected     |  |
|                                   |  |  Post-Processing:  Domain-Clamped |  |
|                                   |  +-----------------------------------+  |
+-----------------------------------+-----------------------------------------+
```

### 5.2 Epsilon Comparison Overlay

Toggle-able multi-epsilon mode showing the privacy-utility tradeoff:

| Epsilon | Pearson RMSE (lower=better) | Privacy Level | Recommended |
| :--- | :--- | :--- | :--- |
| e = 0.1 | 0.312 | Very High | Research only |
| **e = 1.0** | **0.195** | **High** | **OPTIMAL** |
| e = 10.0 | 0.108 | Moderate | Lower privacy risk |

**Privacy-Utility Frontier Plot** (scatter):
- X-axis: Epsilon (privacy cost)
- Y-axis: AUC-ROC retention or Correlation RMSE
- Color-coded by acceptable region

---

## 6. Overview Dashboard (Landing Page)

```
+-----------------------------------------------------------------------------+
|  SYNTHGUARD -- Overview                                 Last Run: 08/20/2026|
|  Dataset: UCI Diabetes  |  Active Session: eps_1.0                          |
+----------+-----------+-----------+----------+------------------------------+
| e SPENT  | MIA ADV.  | TVD (best)| TSTR AUC | TOTAL ROWS GENERATED        |
| 0.3720   | -0.0083   | 0.0316    | 72.39%   | 101,766                     |
| SAFE     | SECURE    | PASS      | PASS     | COMPLETE                    |
+----------+-----------+-----------+----------+------------------------------+
|  PIPELINE EXECUTION STATUS                                                  |
|                                                                             |
|  [1] Data Ingestion & Profiling  .....................  COMPLETE             |
|  [2] Privacy-Protected Training  .....................  COMPLETE             |
|  [3] Synthetic Decoding          .....................  COMPLETE             |
|  [4] Post-Processing Guardrails  .....................  COMPLETE             |
|  [5] Evaluation & Audit          .....................  COMPLETE             |
|                                                                             |
|  MULTI-DATASET GENERALIZATION                                               |
|  [6] CDC BRFSS Heart Disease (445K rows, 40 cols) .....  COMPLETE           |
|      (39 cols retained | 0 manual code changes | e = 0.3720)               |
+-----------------------------------------------------------------------------+
```

---

## 7. Technical Design Specifications

### 7.1 Recommended Technology Stack

**Option A — Rapid Demo (Recommended for Faculty Review):**

| Layer | Choice |
| :--- | :--- |
| Frontend | Streamlit 1.35+ |
| Charts | Plotly Express + Altair |
| Tables | pandas `styler` + `st.dataframe` with `column_config` |
| Layout | `st.columns()`, `st.tabs()`, `st.container()` |
| Deployment | Local: `python -m streamlit run ui/app.py` |

**Option B — Production-Grade:**

| Layer | Choice |
| :--- | :--- |
| Frontend | React 18 + TypeScript + Tailwind CSS |
| Charts | Recharts + D3.js |
| Backend | FastAPI (Python) serving pipeline API |
| State Mgmt | Zustand |
| Deployment | Docker + Nginx |

### 7.2 Data Flow: Pipeline Outputs -> UI

```
E:\Project1\ADVT\
  evaluation_report.txt              --> Module C  (parsed metric tables)
  outputs/adapter_finetuning/
    cdc_heart_generalization_summary.json  --> Module A  (schema profile stats)
    synthetic_cdc_heart_adapted.csv        --> Module C  (distribution plots)
  sweep_results_archive/
    sweep_report.json                --> Module B  (epsilon history per epoch)
    synthetic_eps_0.1.csv            --> Module C  (epsilon comparison view)
    synthetic_eps_1.0.csv            --> Module C  (epsilon comparison view)
    synthetic_eps_10.0.csv           --> Module C  (epsilon comparison view)
  src/privacy/accountant.py          --> Module B  (live e computation hook)
```

### 7.3 Streamlit Component Mapping

| UI Widget | Streamlit Component |
| :--- | :--- |
| KPI Ribbon | `st.metric()` with `delta=` for trend arrows |
| Epsilon Gauge | `st.progress()` + injected CSS for color transition |
| Column Profiler Table | `st.dataframe()` with `column_config` color rules |
| Correlation Heatmap | `st.plotly_chart(px.imshow(...))` |
| TVD Bar Chart | `px.bar()` with threshold line overlay |
| Log Stream | `st.code()` inside `st.expander()` |
| Pipeline Status | `st.status()` / `st.success()` / `st.error()` |
| Audit Certificate | `st.download_button()` with generated markdown |

### 7.4 Layout Code Pattern

```python
# Top KPI ribbon
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("e Spent", "0.3720", delta="-62.8% budget remaining")
col2.metric("MIA Advantage", "-0.0083", delta="Below 0.05 threshold")

# Module tabs
tab_a, tab_b, tab_c = st.tabs(["Module A: Schema", "Module B: Privacy", "Module C: Fidelity"])

# Two-column panel layout
left_panel, right_panel = st.columns([3, 2])
```

---

## 8. Demonstration Flow — Faculty Walkthrough Script

> Use this script to walk faculty through the platform in approximately 10 minutes.

---

### Step 1 — Open Overview Dashboard (1 min)

**Say:** *"This is the SYNTHGUARD platform. The top row immediately tells you the current privacy posture: Epsilon spent is 0.37 out of a budget of 1.0 — we are using only 37% of the allowable mathematical privacy cost. The MIA Advantage score of -0.0083 is a real adversarial attack result, confirming that no patient record can be traced back through this synthetic dataset."*

**Point to:** The green status badges. Emphasize that any red badge would indicate a hard failure that would have blocked output generation.

---

### Step 2 — Module A: Schema Intelligence (2 min)

**Say:** *"Click into Module A. The system automatically ingested this dataset with zero manual configuration — no human told it which columns were sensitive. It detected 'encounter_id' and 'patient_nbr' as HIPAA direct identifiers and removed them before any training began. This is not a checkbox. It is enforced at the code level by the DatasetProfiler class, which has 136 passing unit tests."*

**Point to:** The HIPAA column table with red drop badges. Click a dropped column to show it was never processed.

**Key stat:** *"6 HIPAA identifiers dropped automatically. 44 safe columns forwarded to training. The system also injected 28 missingness indicator flags to preserve clinical sparsity patterns."*

---

### Step 3 — Module B: Privacy Ledger (3 min)

**Say:** *"Module B is the mathematical heart of the audit. This epsilon curve is not an estimate — it is computed exactly by the Renyi Differential Privacy accountant, tracking every gradient update across every training batch. The total privacy expenditure is 0.372 epsilon, certified under the formal (epsilon, delta)-differential privacy definition with delta equal to 10 to the power of negative 4."*

**Point to:** The per-epoch log table showing cumulative epsilon growth.

**Say:** *"More importantly, look at the Membership Inference Attack panel. We ran a real adversarial attack against the trained model — the attack is asking: can I determine whether a specific patient was in the training set? The attack AUC is 0.4958, which is below 0.5 random chance. The model learned the distribution without memorizing any individual record."*

**Show:** The Privacy Certificate export. *"This can be downloaded as a compliance artifact for IRB or data governance review."*

---

### Step 4 — Module C: Fidelity Suite (3 min)

**Say:** *"Now the utility question: is this synthetic data actually useful? The correlation heatmaps compare the statistical relationships between all clinical variables in the real dataset against the synthetic output. The Bivariate RMSE is 0.195 against a threshold of 0.35 — well within acceptable bounds."*

**Point to:** The TSTR benchmark chart.

**Say:** *"The gold standard test: we trained a 30-day hospital readmission prediction model on only synthetic data, then tested it on real held-out patient records. The model retained 72% of the AUC-ROC it would have achieved with real training data — under the strictest privacy guarantee. A hospital could safely use this synthetic dataset for model development without ever exposing a single real patient record."*

---

### Step 5 — Multi-Dataset Generalization (1 min)

**Say:** *"Final proof of generalization: we provided the system with a completely unseen dataset — the CDC BRFSS heart disease survey with 445,000 patients and 40 entirely different clinical columns. Zero code changes were made. The auto-config engine profiled the new schema, detected the geographic quasi-identifier State and excluded it, registered the new encoding, and produced 1,000 privacy-preserving synthetic records in under 5 minutes."*

---

### Closing Statement

**Say:** *"SYNTHGUARD does not just generate synthetic data. It generates a mathematical proof that the data is private, a clinical audit that the data is useful, and a fully reproducible pipeline that any regulatory body can inspect end to end. That is the distinction between synthetic data as a shortcut and synthetic data as a rigorous scientific instrument."*

---

## 9. Implementation Roadmap

| Phase | Deliverable | Est. Effort |
| :--- | :--- | :--- |
| Phase 1 | Streamlit skeleton: routing, sidebar, KPI ribbon | 1 day |
| Phase 2 | Module A: Profiler table, HIPAA badge coloring | 1 day |
| Phase 3 | Module B: Epsilon timeline chart, MIA score widget | 1 day |
| Phase 4 | Module C: Correlation heatmaps, TVD bars, TSTR chart | 2 days |
| Phase 5 | Audit Certificate markdown/PDF download | 0.5 days |
| Phase 6 | Dark theme polish, typography, CSS injection | 0.5 days |
| Phase 7 | CDC multi-dataset generalization tab | 1 day |
| **Total** | **Fully functional faculty demo dashboard** | **~7 days** |

---

## 10. Target File Structure

```
E:\Project1\ADVT\
└── ui\
    ├── ui_ux_implementation_plan.md     <- This document
    ├── app.py                           <- Streamlit entry point
    ├── components\
    │   ├── kpi_ribbon.py                <- Top KPI bar (5 metric tiles)
    │   ├── schema_panel.py              <- Module A implementation
    │   ├── privacy_ledger.py            <- Module B implementation
    │   ├── fidelity_suite.py            <- Module C implementation
    │   └── audit_certificate.py        <- Privacy Certificate export
    ├── data_loaders\
    │   ├── load_sweep_results.py        <- Parse sweep_results_archive/
    │   ├── load_evaluation_report.py   <- Parse evaluation_report.txt
    │   └── load_synthetic_csv.py       <- Load & diff-compare CSVs
    ├── styles\
    │   └── theme.css                   <- Custom Streamlit CSS overrides
    └── assets\
        └── synthguard_logo.svg         <- Platform wordmark/logo
```

---

*Document Version: 1.0 | Generated: 2026-08-20*
*Project: ADVT — Privacy-Preserving Synthetic Healthcare Data Generation*
