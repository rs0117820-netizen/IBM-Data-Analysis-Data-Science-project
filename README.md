# HealthFlow

## Hospital Patient Flow Analytics & Waiting-Time Prediction
> **Turning Patient Flow Data into Actionable Healthcare Insights**

**Student Name:** Rohan  
**Project:** HealthFlow  
**Academic Program:** BharatCares / AICTE Data Analytics with AI Academic Internship  
**Submission Package:** Google Form Upload (4 Submission Files Only)  

---

### Google Form Submission Files (4 Files Only)

This submission is 100% self-contained within the following **four Google Form submission files**. Evaluators do **not** need to download, create, or configure any external folders (`src/`, `data/`, `models/`) or standalone runner scripts (`healthflow.py`).

| # | Submitted File Name | Description & Submission Role |
| :-: | :--- | :--- |
| **1** | [`Rohan_HealthFlow.ipynb`](./Rohan_HealthFlow.ipynb) | **Single Complete Code Submission**: Contains the entire end-to-end implementation including backend data ingestion, sentinel data cleaning, leak-free feature engineering, exploratory data analysis, machine learning model training/benchmarking, and the complete interactive frontend clinical decision support dashboard with enterprise CSS styling. |
| **2** | [`requirements.txt`](./requirements.txt) | **Project Dependencies**: Exact third-party Python package specifications required to reproduce the environment. |
| **3** | [`Rohan_HealthFlowProjectReport.docx`](./Rohan_HealthFlowProjectReport.docx) | **Formal Academic Project Report**: Comprehensive project documentation in Microsoft Word format covering executive summary, clinical context, exploratory analysis, ML benchmarks, and operational recommendations. |
| **4** | [`README.md`](./README.md) | **Project Documentation & Execution Guide**: System architecture, benchmark results, dataset description, and step-by-step evaluation instructions. |

> [!NOTE]
> **Self-Contained Code Guarantee**: `Rohan_HealthFlow.ipynb` is the sole, comprehensive code artifact. All backend algorithms (data processing, feature derivation, model training) and frontend components (Plotly interactive charts, Light/Dark styling tokens, and the 7-page Streamlit clinical decision support dashboard) are implemented directly inside this single notebook.

---

## Project Overview

**HealthFlow** is an end-to-end healthcare analytics and machine learning application that analyzes real-world hospital emergency patient-flow telemetry, identifies queue bottlenecks and department congestion patterns, and deploys predictive regression models to estimate patient waiting times without data leakage.

Built upon **66,197 operational telemetry records** (59,663 cleaned, valid observations after removing 6,534 sentinel records) collected across **18 Canadian hospital emergency departments and urgent care centres** in the Alberta Health Services (AHS) network, HealthFlow equips healthcare operational managers with empirical visibility into capacity surges, 24-hour diurnal load curves, and facility disparities.

---

## Problem Statement

> *"How can hospital patient-flow data be analyzed to identify waiting-time patterns, workload patterns, department congestion, and other operational factors, and how can machine learning be used to estimate expected patient waiting time?"*

Hospital emergency departments frequently experience sudden surges in patient arrivals and severe queue bottlenecks. Prolonged waiting times degrade patient satisfaction, increase ambulance offload delays, and strain clinical personnel. HealthFlow addresses this operational challenge by replacing static intuition with empirical telemetry analytics and predictive ML modeling.

---

## Objectives

1. **Authentic Telemetry Ingestion**: Ingest exactly **66,197 operational records** from Alberta Health Services provincial health authorities without synthetic or fabricated data.
2. **Disciplined Data Cleaning**: Detect and filter **6,534 closed/offline sentinel telemetry records** (`waitTime < 0`), deduplicate observation timestamps, and retain **59,663 verified, high-integrity observations**.
3. **Leakage-Free Feature Engineering**: Formulate temporal variables, facility tiers, health zones, and concurrent system load strictly available at the time of patient arrival.
4. **Exploratory Data Analysis**: Quantify 24-hour diurnal curves, day-of-week dynamics, facility tier disparities, and waiting-time percentile distributions.
5. **Machine Learning Model Benchmarking**: Train and compare Ridge Regression, Random Forest, and HistGradientBoosting regressors using a strict chronological 80/20 train/test split (47,730 train vs 11,933 test records).
6. **Interactive Clinical Dashboard**: Deliver an enterprise clinical dashboard with 7 comprehensive analytical pages and real-time waiting-time prediction.
7. **Actionable Operational Insights**: Synthesize evidence-based recommendations regarding clinical shift alignment, regional patient diversion, and queue escalation triggers.

---

## Key Features

- **End-to-End Self-Contained Pipeline**: Ingestion, quality auditing, sentinel removal (6,534 records filtered), deduplication, and chronological feature derivation executed entirely within `Rohan_HealthFlow.ipynb`.
- **Leakage-Free Predictive Engine**: Trained strictly on arrival-time context (hospital, hour, day, month, facility tier, health zone, concurrent active facilities, and concurrent system wait time).
- **Interactive Multi-Page Frontend Dashboard**: 7 comprehensive modules covering executive KPIs, longitudinal volume trends, queue analytics, scenario prediction, data exploration, and model transparency.
- **Benchmarked Regression Architecture**: Ridge regression baseline, Random Forest ensemble, and champion HistGradientBoosting model (Test MAE: 33.74 min, Test R²: 0.4293).
- **Unified 4-File Submission**: Everything needed to evaluate, reproduce, and review the project is packaged into the 4 Google Form submission files.

---

## Dataset Description

The dataset monitors high-frequency emergency department operational wait-time telemetry logged at ~5-minute intervals across 18 healthcare centres over a continuous 6-week operational window (August 24, 2018 to October 04, 2018).

- **Primary Source Authority**: Alberta Health Services (AHS), Edmonton & Calgary Zones, Alberta, Canada
- **Curated / Archival Repository**: S. R. Veale Open Healthcare Telemetry ([srveale/emergency-wait-times](https://github.com/srveale/emergency-wait-times))
- **Direct Source Data URL**: `https://raw.githubusercontent.com/srveale/emergency-wait-times/master/EWT_DATA.csv`
- **Data Compliance**: 100% de-identified public sector operational transparency telemetry. Contains zero Protected Health Information (PHI).

| Attribute | Value / Specification |
| :--- | :--- |
| **Total Raw Records** | 66,197 observations |
| **Total Cleaned Records** | 59,663 observations (after removing 6,534 sentinel records with `waitTime < 0` and duplicates) |
| **Raw Schema Columns** | 6 columns (`Unnamed: 0`, `waitTime`, `hospitalName`, `date`, `epochTime`, `ID`) |
| **Target Variable** | `waitTime` (continuous minutes to evaluate by an emergency physician) |
| **Monitored Facilities** | 18 hospital emergency departments and urgent care centres |
| **Health Zones** | 4 geographical zones (Calgary Zone, Edmonton Zone, Central Zone, South Zone) |

### Monitored Healthcare Facilities by Operational Tier

1. **Pediatric Emergency Departments**: Alberta Children's Hospital, Stollery Children's Hospital
2. **Academic Tertiary Trauma Centres**: Foothills Medical Centre, University of Alberta Hospital, Royal Alexandra Hospital
3. **Acute Urban General Hospitals**: Peter Lougheed Centre, Rockyview General Hospital, South Health Campus, Misericordia Community Hospital, Grey Nuns Community Hospital
4. **Community Hospitals & Ambulatory Care**: Sturgeon Community Hospital, Fort Sask Community Hospital, Leduc Community Hospital, Strathcona Community Hospital, WestView Health Centre, Northeast Community Health Centre
5. **Regional & Rural Health Centres**: Chinook Regional Hospital, Medicine Hat Regional Hospital, Lacombe Hospital and Care Centre, Innisfail Health Centre

---

## System Architecture

All operational, analytical, modeling, and presentation layers are fully unified inside **`Rohan_HealthFlow.ipynb`**:

```
HealthFlow Unified Operational Architecture (Inside Rohan_HealthFlow.ipynb)
┌────────────────────────────────────────────────────────────────────────┐
│                      Raw Operational Telemetry                         │
│            (Alberta Health Services 66,197 raw records)                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             Stage 1: Telemetry Ingestion & Schema Audit                │
│       • Ingests 66,197 records directly from remote / local cache      │
│       • Validates columns, timestamps, and facility identifiers        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             Stage 2: Data Cleaning & Sentinel Filtering                │
│       • Detects & filters 6,534 offline sentinel records (waitTime < 0)│
│       • Trims whitespace, parses datetimes, deduplicates timestamps    │
│       • Retains exactly 59,663 verified, high-integrity records        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             Stage 3: Leakage-Free Feature Engineering                  │
│       • Temporal: Hour of Day, Day of Week, Month, Weekend, Peak Flag  │
│       • Facility Taxonomy: 5 Operational Tiers, 4 Health Zones         │
│       • Concurrency Load: Active Facilities & System Concurrent Average│
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│  Chronological Train/Test Split │   │   Exploratory Data Analytics    │
│  • 80% Train (47,730 records)   │   │  • 24-Hour Diurnal Curves       │
│  • 20% Test (11,933 records)    │   │  • Workload & Facility Rankings │
│  • Strict zero-lookahead split  │   │  • 10th-90th Percentile Bands   │
└────────────────┬────────────────┘   └────────────────┬────────────────┘
                 ▼                                     ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│    Machine Learning Engine      │   │   Frontend Clinical Dashboard   │
│  • Ridge Regression (Baseline)  │   │  (Self-Contained in Notebook)   │
│  • Random Forest Regressor      │   │  • 1. Executive Overview & KPIs │
│  • HistGradientBoosting (Best)  │──▶│  • 2. Patient Flow Analytics    │
│  • Leakage-free arrival context │   │  • 3. Waiting Time Analytics    │
│  • MAE, RMSE, R² Benchmarking   │   │  • 4. Scenario Prediction Tool  │
└─────────────────────────────────┘   │  • 5. Operational Insights      │
                                      │  • 6. Telemetry Data Explorer   │
                                      │  • 7. Model Architecture Info   │
                                      └─────────────────────────────────┘
```

---

## End-to-End Pipeline in `Rohan_HealthFlow.ipynb`

The single notebook submission executes the complete six-stage analytical workflow sequentially:

1. **Telemetry Ingestion Stage (Section 3)**:
   - Ingests the 66,197 operational records from the verified Alberta Health Services dataset.
   - Validates data completeness and schema attributes.
2. **Data Cleaning & Sentinel Filtering Stage (Section 4)**:
   - Audits missing values (0 nulls in primary identifiers).
   - Identifies and removes **6,534 sentinel records** where `waitTime < 0` (indicating sensor offline or facility closed periods).
   - Deduplicates timestamp-facility pairs and trims whitespace.
   - Retains **59,663 clean observations** with waiting times ranging from 0 to 582 minutes.
3. **Leakage-Free Feature Engineering Stage (Section 5)**:
   - Formulates temporal cyclical features: `hour`, `day_of_week`, `day_name`, `month`, `is_weekend`, `peak_hour_flag` (11:00–21:00).
   - Enriches records with facility operational tiers and provincial health zones.
   - Calculates concurrent system load (number of concurrently reporting facilities and concurrent system-wide average wait time) using only arrival-time information.
4. **Exploratory Data Analysis Stage (Sections 6–7)**:
   - Calculates statistical distribution metrics and generates interactive Plotly visualizations directly inline.
   - Analyzes diurnal hourly patterns, weekday/weekend dynamics, and facility tier disparities.
5. **Machine Learning Model Training & Evaluation Stage (Sections 8–11)**:
   - Implements a strict chronological 80/20 train/test split (47,730 training vs 11,933 test observations).
   - Builds preprocessing pipelines with `StandardScaler` and `OneHotEncoder`.
   - Trains and evaluates Ridge Regression, Random Forest, and HistGradientBoosting regressors.
6. **Frontend Clinical Decision Support Dashboard Stage (Section 12)**:
   - Consolidates the complete UI design system, Light/Dark styling tokens, and 7-page interactive dashboard.

---

## Exploratory Data Analysis Results

Key statistical metrics computed across the 59,663 valid telemetry observations:

| Operational Metric | Empirical Value | Healthcare Interpretation |
| :--- | :---: | :--- |
| **Total Valid Observations** | **59,663 records** | Complete telemetry monitoring across 18 hospitals |
| **System Mean Wait Time** | **94.4 minutes** | Typical patient waiting time across all 18 facilities |
| **System Median Wait Time** | **79.0 minutes** | 50% of patient visits are seen within 79 minutes |
| **Standard Deviation** | **63.4 minutes** | High variability driven by facility tiers and arrival hour |
| **Interquartile Range (IQR)** | **86.0 minutes** | Spread between 25th percentile (46m) and 75th percentile (132m) |
| **90th Percentile Delay** | **183.0 minutes** | Critical tail delay; 10% of arrivals wait over 3 hours |
| **Diurnal Peak Arrival Hour** | **14:00 – 16:00** | Afternoon surge window with maximum concurrent load |
| **Diurnal Peak Wait Hour** | **17:00 – 19:00** | Lagged peak delay (~118 minutes) as arrivals accumulate |
| **Diurnal Nadir Wait Hour** | **05:00 – 07:00** | Early morning lull (average wait drops to ~52 minutes) |
| **Weekday Mean Wait Time** | **96.1 minutes** | Higher queue delay driven by outpatient consultations |
| **Weekend Mean Wait Time** | **90.2 minutes** | Slightly lower mean wait time across urban network |

---

## Machine Learning Modeling

### Supervised Formulation
Given patient arrival context at time $t$:
$$\mathbf{x} = [\text{facility}, \text{facility\_tier}, \text{health\_zone}, \text{hour}, \text{day\_of\_week}, \text{month}, \text{is\_weekend}, \text{concurrent\_active\_facilities}, \text{concurrent\_system\_load}, \text{peak\_hour\_flag}]$$

Predict continuous waiting time in minutes:
$$\hat{y} = f(\mathbf{x})$$

### Safeguards Against Data Leakage
- **Chronological Holdout Split**: The dataset is strictly ordered by timestamp. The first **80% (47,730 records)** are used for training; the final **20% (11,933 records)** are reserved strictly for testing.
- **Arrival-Time Information Boundary**: No post-triage or discharge variables (consultation duration, admission status, disposition) are used as predictors.

---

## Model Evaluation & Benchmarks

All models were evaluated on the held-out test set (11,933 future observations). **All metrics are genuine, reproducible, and computed directly within the notebook:**

| Model Architecture | Train MAE | Test MAE | Train RMSE | Test RMSE | Train R² | Test R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression (Baseline)** | 36.82 m | 35.02 m | 49.24 m | 47.11 m | 0.4114 | 0.3795 |
| **Random Forest Regressor** | 21.16 m | 35.21 m | 28.43 m | 47.25 m | 0.8038 | 0.3756 |
| **HistGradientBoosting Regressor (Champion)** | **29.82 m** | **33.74 m** | **39.51 m** | **45.18 m** | **0.6210** | **0.4293** |

### Benchmark Takeaways
- **HistGradientBoostingRegressor** achieved superior generalization on unseen forward telemetry, with a **Test MAE of 33.74 minutes** and **Test R² of 0.4293**.
- Non-linear tree boosting effectively captured complex interaction effects between arrival hour, facility tier, and concurrent regional workload without target leakage.

---

## Frontend Clinical Decision Support Dashboard

The complete frontend application code, UI styling, and visualization modules are embedded directly within Section 12 of [`Rohan_HealthFlow.ipynb`](./Rohan_HealthFlow.ipynb), featuring:

1. **Overview & Executive KPIs**: High-level metrics (Total Observations, Mean Wait, Median Wait, Peak Surge Hour, Most Congested Facility), 24-hour diurnal profile, and department workload rankings.
2. **Patient Flow Analytics**: Longitudinal daily telemetry volume trends, 7-day moving averages, day-of-week breakdown, and comprehensive department metrics tables.
3. **Waiting Time Analytics**: Queue delay analysis with 10th-to-90th percentile diurnal delay bands, waiting time dispersion histograms, and facility tier disparities.
4. **AI/ML Scenario Prediction Engine**: Interactive clinical calculator allowing users to select facility, arrival hour, day of week, and system congestion level, generating predicted waiting times, confidence intervals, and clinical disclaimers.
5. **Operational Insights**: Evidence-based findings organized by Temporal Patterns, Facility Taxonomy, Day-of-Week Dynamics, and Operational Risk with actionable clinical management recommendations.
6. **Data Explorer**: Searchable, filterable interactive table with dynamic facility and date filters and instant CSV export.
7. **Model Information & Transparency**: Architecture specifications, benchmark comparison tables, feature importance rankings, and data leakage prevention safeguards.

---

## Technologies Used

- **Python 3.10+ / 3.12**
- **Pandas** (>=2.0.0) — High-performance data manipulation and cleaning
- **NumPy** (>=1.24.0) — Numerical operations and array transformations
- **Scikit-Learn** (>=1.3.0) — Preprocessing, Ridge, Random Forest, and HistGradientBoosting regressors
- **Matplotlib** (>=3.7.0) — Inline analytical visualizations
- **Plotly** (>=5.15.0) — Interactive charts and percentile delay bands
- **Streamlit** (>=1.25.0) — Clinical decision support dashboard UI framework
- **Joblib** (>=1.3.0) — Model pipeline serialization
- **Jupyter Notebook** — Interactive notebook environment

---

## Submission Package Details

The project is packaged strictly into the **four Google Form submission files**:

```
HealthFlow Submission Package (Google Form Files):
│
├── Rohan_HealthFlow.ipynb             # Single complete code submission (Backend pipeline, ML models, Frontend dashboard)
├── requirements.txt                   # Exact Python package dependencies
├── Rohan_HealthFlowProjectReport.docx # Comprehensive Academic Project Report (Microsoft Word)
└── README.md                          # Comprehensive project documentation & execution guide
```

### File Breakdown:
1. **`Rohan_HealthFlow.ipynb`**:
   - The single complete code submission file containing both backend and frontend.
   - Houses data ingestion, sentinel filtering, feature engineering, statistical EDA, ML training and benchmarking, and the full multi-page Streamlit clinical dashboard code.
2. **`requirements.txt`**:
   - Lists the exact third-party Python packages required to run the code.
3. **`Rohan_HealthFlowProjectReport.docx`**:
   - The formal academic internship project report detailing clinical motivation, methodology, experimental results, and operational insights.
4. **`README.md`**:
   - Complete project documentation, operational architecture, benchmark results, and execution guide.

---

## Execution Guide

### 1. Set Up Environment & Install Dependencies
Open a terminal in the folder containing the submission files, create a virtual environment, and install dependencies:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Run the Complete Project Notebook (`Rohan_HealthFlow.ipynb`)
Open and execute `Rohan_HealthFlow.ipynb` using Jupyter Notebook, JupyterLab, VS Code, or Google Colab:

```bash
jupyter notebook Rohan_HealthFlow.ipynb
```

In the notebook interface, click **Kernel -> Restart & Run All** (or **Run All** in VS Code).

#### What Executes During Notebook Run:
- **Sections 1–2**: Loads libraries, sets up logging, and configures healthcare facility taxonomies and health zones.
- **Section 3**: Ingests and audits the 66,197 operational records from the authentic Alberta Health Services telemetry dataset.
- **Section 4**: Cleans data and removes 6,534 closed/offline sentinel records (`waitTime < 0`), deduplicating timestamps and retaining 59,663 valid observations.
- **Section 5**: Derives leakage-free temporal and concurrent workload features available at patient arrival.
- **Sections 6–7**: Computes statistical distributions and displays interactive Plotly charts directly inline (diurnal profiles, percentiles, facility workload).
- **Sections 8–11**: Performs chronological 80/20 train/test split, trains Ridge, Random Forest, and HistGradientBoosting models, and prints evaluation metrics (MAE, RMSE, R²).
- **Section 12**: Contains the complete, self-contained interactive Streamlit Clinical Decision Support Dashboard application code.

---

## Key Operational Insights & Recommendations

All findings are derived directly from actual telemetry data:

1. **Diurnal Congestion Cycle**: Waiting times surge from 12:00 to 22:00 daily, peaking in late afternoon (~118 minutes).  
   *Recommendation*: Stagger emergency clinical staffing shifts to overlap with the afternoon/evening surge window rather than traditional static 8-hour shifts.
2. **Facility Tier Disparities**: Academic tertiary trauma centres experience average waiting times exceeding 115–125 minutes, whereas community clinics average under 50–60 minutes.  
   *Recommendation*: Implement regional patient transit advisories to divert non-emergent ambulatory patients (CTAS 4–5) to nearby community urgent care clinics.
3. **Day-of-Week Load Dynamics**: Weekday waiting times average 96.1 minutes compared to 90.2 minutes on weekends, with mid-week surges driven by delayed primary care access.  
   *Recommendation*: Schedule elective outpatient follow-ups away from peak weekday surge days to mitigate bed boarding in emergency observation units.
4. **Queue Tail Risk (90th Percentile Delay)**: While median wait time is 79.0 minutes, the 90th percentile wait reaches 183.0 minutes, with peak facilities exceeding 4 hours.  
   *Recommendation*: Deploy automated operational alerts when facility waiting times exceed 150 minutes to activate rapid medical evaluation (RME) pods and fast-track discharge protocols.

---

## Limitations

- **Triage Acuity Priority**: Public telemetry records operational waiting times for mid-acuity arrivals. Patients presenting with immediately life-threatening conditions (resuscitation CTAS 1) bypass queue waiting times and are treated immediately.
- **Regional Model Specifics**: The data reflects Canadian publicly funded emergency care in Alberta; transferability to other healthcare structures requires local operational recalibration.
- **Operational Scope**: HealthFlow is an administrative analytics and queue management platform; it does not offer clinical diagnosis, triage classification, or patient treatment advice.

---

## Future Improvements

1. **Environmental Telemetry**: Integrate ambient weather data (blizzards, extreme cold) to analyze seasonal influxes.
2. **FHIR / EHR Interoperability**: Stream live clinical queue metrics using HL7/FHIR interfaces.
3. **Queue-Length Simulation**: Implement SimPy discrete-event modeling to simulate hospital bed occupancy and physician service times.
4. **Cloud Scalability**: Containerized deployment for hospital network integration.

---

## Disclaimer

> **IMPORTANT**: This project is intended for educational and analytical purposes. It does not provide medical diagnosis, clinical advice, emergency triage decisions, or treatment recommendations. In any medical emergency, patients should call emergency services or proceed immediately to the nearest hospital emergency department.
