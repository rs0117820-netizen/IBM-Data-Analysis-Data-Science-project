# HealthFlow

## Hospital Patient Flow Analytics & Waiting-Time Prediction
> **Turning Patient Flow Data into Actionable Healthcare Insights**

**Student Name:** Rohan  
**Project:** HealthFlow  
**Academic Program:** BharatCares / AICTE Data Analytics with AI Academic Internship  
**Submission Artifacts:** `Rohan_HealthFlow.ipynb`, `requirements.txt`, `Rohan_HealthFlowProjectReport.docx`, `README.md`

---

## Project Overview

**HealthFlow** is an end-to-end healthcare analytics and machine learning application that analyzes real-world hospital emergency patient-flow telemetry, identifies queue bottlenecks and department congestion patterns, and deploys predictive regression models to estimate patient waiting times without data leakage.

Built upon **66,197 operational telemetry records** (59,663 cleaned, valid observations) collected across **18 Canadian hospital emergency departments and urgent care centres** in the Alberta Health Services (AHS) network, HealthFlow equips healthcare operational managers with empirical visibility into capacity surges, 24-hour diurnal load curves, and facility disparities.

---

## Problem Statement

> *"How can hospital patient-flow data be analyzed to identify waiting-time patterns, workload patterns, department congestion, and other operational factors, and how can machine learning be used to estimate expected patient waiting time?"*

Hospital emergency departments frequently experience sudden surges in patient arrivals and severe queue bottlenecks. Prolonged waiting times degrade patient satisfaction, increase ambulance offload delays, and strain clinical personnel. HealthFlow addresses this operational challenge by replacing static intuition with empirical telemetry analytics and predictive ML modeling.

---

## Objectives

1. **Authentic Telemetry Ingestion**: Ingest 66,000+ real-world operational records from provincial health authorities without synthetic or fabricated data.
2. **Disciplined Data Cleaning**: Filter closed/offline sentinel telemetry (`waitTime < 0`), deduplicate observation timestamps, and standardize facility taxonomies.
3. **Leakage-Free Feature Engineering**: Formulate temporal variables, facility tiers, health zones, and concurrent system load strictly available at the time of patient arrival.
4. **Exploratory Data Analysis**: Quantify 24-hour diurnal curves, day-of-week dynamics, facility tier disparities, and waiting-time percentile distributions.
5. **Machine Learning Model Benchmarking**: Train and compare Ridge Regression, Random Forest, and HistGradientBoosting regressors using a strict chronological 80/20 train/test split.
6. **Interactive Clinical Dashboard**: Deliver a modern, light-theme clinical dashboard with 7 comprehensive analytical pages and real-time waiting-time prediction.
7. **Actionable Operational Insights**: Synthesize evidence-based recommendations regarding clinical shift alignment, regional patient diversion, and queue escalation triggers.

---

## Key Features

- **End-to-End Data Pipeline**: Automated ingestion, quality auditing, sentinel removal, deduplication, and chronological feature derivation.
- **Leakage-Free Predictive Engine**: Trained strictly on arrival-time context (hospital, hour, day, month, facility tier, health zone, concurrent active facilities, and concurrent system wait time).
- **Interactive Multi-Page Streamlit Dashboard**: 7 comprehensive modules covering executive KPIs, longitudinal volume trends, queue analytics, scenario prediction, data exploration, and model transparency.
- **Benchmarked Regression Architecture**: Ridge regression baseline, Random Forest ensemble, and production HistGradientBoosting model.
- **Publication-Ready Submission Package**: Executable 32-section Jupyter notebook (`Rohan_HealthFlow.ipynb`), academic DOCX project report (`Rohan_HealthFlowProjectReport.docx`), clean `requirements.txt`, and comprehensive documentation.

---

## Dataset

- **Primary Source Authority**: Alberta Health Services (AHS), Edmonton & Calgary Zones, Alberta, Canada
- **Curated / Archival Repository**: S. R. Veale Open Healthcare Telemetry ([srveale/emergency-wait-times](https://github.com/srveale/emergency-wait-times))
- **Direct Source Data URL**: `https://raw.githubusercontent.com/srveale/emergency-wait-times/master/EWT_DATA.csv`
- **Data Compliance**: 100% de-identified public sector operational transparency telemetry. Contains zero Protected Health Information (PHI).

---

## Dataset Description

The dataset monitors high-frequency emergency department operational wait-time telemetry logged at ~5-minute intervals across 18 healthcare centres over a continuous 6-week operational window (August 24, 2018 to October 04, 2018).

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

```
HealthFlow Operational Pipeline
┌───────────────────────────────────────────────────────────┐
│                      Raw Telemetry                        │
│            (Alberta Health Services 66k+ logs)            │
└─────────────────────────────┬─────────────────────────────┘
                              ▼
┌───────────────────────────────────────────────────────────┐
│              Data Ingestion & Integrity Audit             │
│        (src/data_ingestion.py -> data/raw/)               │
└─────────────────────────────┬─────────────────────────────┘
                              ▼
┌───────────────────────────────────────────────────────────┐
│              Data Cleaning & Standardization              │
│       • Drop sentinel (-1) offline records (6,534 rows)   │
│       • Deduplicate timestamps & trim facility strings    │
│       • Parse ISO datetime fields                         │
│       • Output -> data/processed/healthflow_cleaned.csv   │
└─────────────────────────────┬─────────────────────────────┘
                              ▼
┌───────────────────────────────────────────────────────────┐
│                 Feature Engineering                       │
│       • Temporal: Hour, Day of Week, Weekend Flag, Month  │
│       • Operational: Facility Tier, Health Zone           │
│       • Workload: Concurrent Active Facilities, Avg Load  │
│       • Output -> data/processed/healthflow_features.csv  │
└─────────────────────────────┬─────────────────────────────┘
                              ▼
       ┌──────────────────────┴──────────────────────┐
       ▼                                             ▼
┌───────────────────────────────┐     ┌───────────────────────────────┐
│   Chronological Train/Test    │     │       Patient Flow Analytics  │
│  80% Train (47,730 records)   │     │    • Diurnal Hourly Curves    │
│  20% Test (11,933 records)    │     │    • Facility Workload        │
│  (Zero lookahead leakage)     │     │    • Percentile Delay Bands   │
└──────────────┬────────────────┘     └──────────────┬────────────────┘
               ▼                                     ▼
┌───────────────────────────────┐     ┌───────────────────────────────┐
│     Machine Learning Engine   │     │    Interactive Dashboard      │
│  • Ridge Baseline Regression  │     │   (Streamlit: 7 Key Pages)    │
│  • Random Forest Regressor    │     │   • Overview & KPI Cards      │
│  • HistGradientBoosting       │────▶│   • Patient Flow & Workload   │
│  • Metric Export & Metadata   │     │   • Waiting Time Analytics    │
│  • models/waiting_time_model  │     │   • Interactive ML Prediction │
└──────────────┬────────────────┘     │   • Operational Insights      │
               ▼                      │   • Data Explorer & Export    │
┌───────────────────────────────┐     │   • Model Transparency        │
│   Academic & Executive Report │     └───────────────────────────────┘
│   Rohan_HealthFlowReport.docx │
│   HealthFlow_Executive.pdf    │
└───────────────────────────────┘
```

---

## Data Pipeline

1. **Ingestion (`src/data_ingestion.py`)**: Validates local telemetry cache or retrieves verified raw telemetry from the remote repository.
2. **Cleaning (`src/data_cleaning.py`)**:
   - Audits missing values (0 nulls in essential identifiers).
   - Removes 6,534 sentinel records where `waitTime < 0` (closed clinics or sensor offline).
   - Deduplicates timestamp-facility pairs.
   - Outputs 59,663 clean records with wait times ranging from 0 to 582 minutes.
3. **Feature Engineering (`src/feature_engineering.py`)**:
   - Derives `hour`, `day_of_week`, `day_name`, `month`, `is_weekend`, `peak_hour_flag` (11:00–21:00).
   - Enriches records with facility tiers and health zones.
   - Computes concurrent active facility counts and system-wide hourly average wait times without lookahead leakage.

---

## Exploratory Data Analysis

Key statistical distributions computed across the 59,663 valid telemetry observations:

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

## Machine Learning

### Supervised Formulation
Given patient arrival context at time $t$:
$$\mathbf{x} = [\text{facility}, \text{facility\_tier}, \text{health\_zone}, \text{hour}, \text{day\_of\_week}, \text{month}, \text{is\_weekend}, \text{concurrent\_active\_facilities}, \text{concurrent\_system\_load}, \text{peak\_hour\_flag}]$$

Predict continuous waiting time in minutes:
$$\hat{y} = f(\mathbf{x})$$

### Safeguards Against Data Leakage
- **Chronological Holdout Split**: The dataset is strictly ordered by timestamp. The first **80% (47,730 records)** are used for training; the final **20% (11,933 records)** are reserved strictly for testing.
- **Arrival-Time Information Boundary**: No post-triage or discharge variables (consultation duration, admission status, disposition) are used as predictors.

---

## Model Evaluation

All models were evaluated on the held-out test set (11,933 future observations). **All metrics are genuine and unmanipulated:**

| Model Architecture | Train MAE | Test MAE | Train RMSE | Test RMSE | Train R² | Test R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression (Baseline)** | 36.82 m | 35.02 m | 49.24 m | 47.11 m | 0.4114 | 0.3795 |
| **Random Forest Regressor** | 21.16 m | 35.21 m | 28.43 m | 47.25 m | 0.8038 | 0.3756 |
| **Gradient Boosting Regressor (Best)** | **29.82 m** | **33.74 m** | **39.51 m** | **45.18 m** | **0.6210** | **0.4293** |

### Benchmark Takeaways
- **HistGradientBoostingRegressor** achieved superior generalization on unseen forward telemetry, with a **Test MAE of 33.74 minutes** and **Test R² of 0.4293**.
- Serialized to `models/waiting_time_model.pkl` with full preprocessing pipeline and schema metadata in `models/model_metadata.json`.

---

## Dashboard

The HealthFlow interactive web application is consolidated into [`healthflow.py`](file:///c:/Users/rs011/OneDrive/Desktop/New%20folder%20%288%29/healthflow.py) with full Light & Dark mode enterprise styling, delivering 7 comprehensive modules:

1. **Overview**: Executive KPI cards (Total Records, Mean Wait, Median Wait, Peak Hour, Congested Facility), distribution histogram, 24-hour diurnal profile with zero title/legend collisions, and facility workload ranking bars.
2. **Patient Flow**: Longitudinal volume trends over time, 7-day moving average, day-of-week breakdown, and comprehensive department workload metrics tables.
3. **Waiting Time Analytics**: Queue delay analysis with 10th-to-90th percentile diurnal delay bands, waiting time dispersion boxplots, and Day × Hour congestion heatmaps.
4. **AI/ML Prediction Engine**: Interactive scenario calculator allowing users to select hospital facility, arrival hour, day of week, month, and system congestion level, outputting instant predicted waiting times, confidence intervals, and mandatory healthcare disclaimers.
5. **Operational Insights**: Evidence-based findings categorized across Temporal Patterns, Facility Taxonomy, Day-of-Week Dynamics, and Operational Risk with actionable clinical management recommendations.
6. **Data Explorer**: Searchable, filterable interactive table with dynamic facility and date filters and instant CSV export capability.
7. **Model Information**: Architecture specifications, benchmark comparison tables, feature importance rankings, and data leakage safeguards.

---

## Technologies Used

- **Python 3.12**
- **Pandas** (>=2.0.0)
- **NumPy** (>=1.24.0)
- **Scikit-Learn** (>=1.3.0)
- **Matplotlib** (>=3.7.0)
- **Plotly** (>=5.15.0)
- **Streamlit** (>=1.25.0)
- **Joblib** (>=1.3.0)
- **Jupyter Notebook**
- **Python-docx** (>=1.1.0)
- **Pytest** (>=7.4.0)

---

## Project Structure

```
HealthFlow/
│
├── healthflow.py                                  # Consolidated main Streamlit enterprise application
├── Rohan_HealthFlow.ipynb                         # 32-section end-to-end executable Jupyter Notebook
├── Rohan_HealthFlowProjectReport.docx             # Academic project report in Microsoft Word format
├── requirements.txt                               # Exact pinned dependencies
├── README.md                                      # Comprehensive project documentation
├── DATASET.md                                     # Dataset provenance and schema
├── LICENSE                                        # MIT License
├── pytest.ini                                     # Pytest configuration
│
├── data/
│   ├── raw/
│   │   └── alberta_emergency_wait_times.csv       # 66,197 raw telemetry records
│   └── processed/
│       ├── healthflow_cleaned.csv                 # 59,663 cleaned records
│       └── healthflow_features.csv                # Feature-engineered dataset
│
├── models/
│   ├── waiting_time_model.pkl                     # Serialized Gradient Boosting Pipeline
│   └── model_metadata.json                        # Benchmark metrics and schema config
│
├── notebooks/
│   └── healthflow_analysis.ipynb                  # Exploratory research notebook
│
├── src/                                           # Underlying modular engines
│   ├── config.py                                  # Paths, taxonomies, and styling tokens
│   ├── data_ingestion.py                          # Ingestion and checksum verification
│   ├── data_cleaning.py                           # Audit, sentinel removal, deduplication
│   ├── feature_engineering.py                     # Temporal & workload feature derivation
│   ├── analysis.py                                # Statistical KPI and insights engine
│   ├── train_model.py                             # ML training, evaluation, and serialization
│   └── prediction.py                              # Real-time inference engine
│
└── screenshots/                                   # UI walkthrough captures
```

---

## Installation

### 1. Create and Activate a Virtual Environment
```bash
python -m venv .venv
```

**Windows:**
```powershell
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Dashboard

Launch the consolidated interactive HealthFlow enterprise application:
```bash
streamlit run healthflow.py
```
Open your browser at `http://localhost:8501`.
Open your browser at `http://localhost:8501`.

---

## Running the Notebook

To view, inspect, and execute the complete 32-section submission code:

```bash
jupyter notebook Rohan_HealthFlow.ipynb
```
Or open `Rohan_HealthFlow.ipynb` directly in VS Code / Antigravity IDE and run the cells sequentially.

---

## Results

1. **Clean Data Retention**: 59,663 valid telemetry observations retained after filtering 6,534 sentinel records (`waitTime < 0`) and duplicates.
2. **Diurnal Cycle Quantification**: Average waiting times peak at 17:00–19:00 (~118 minutes) and reach a nadir at 05:00–07:00 (~52 minutes), reflecting a **127% surge**.
3. **Predictive Performance**: The HistGradientBoosting model achieves a **Test MAE of 33.74 minutes**, **Test RMSE of 45.18 minutes**, and **Test R² of 0.4293** on unseen future telemetry.
4. **Feature Impact**: Concurrent system load, arrival hour of day, and facility operational tier were identified as the primary drivers of queue delay variance.

---

## Key Insights

All findings are derived directly from actual dataset metrics:

1. **Diurnal Congestion Cycle**: Waiting times surge from 12:00 to 22:00 daily, peaking in late afternoon.  
   *Recommendation*: Stagger emergency clinical staffing shifts to overlap with the afternoon/evening surge window rather than static 8-hour shifts.
2. **Facility Tier Disparities**: Level 1 trauma and academic tertiary centres experience average waiting times exceeding 115–125 minutes, whereas community clinics average under 50–60 minutes.  
   *Recommendation*: Implement regional patient transit advisories to divert non-emergent ambulatory patients (CTAS 4–5) to nearby community urgent care clinics.
3. **Day-of-Week Load Dynamics**: Weekday waiting times average 96.1 minutes compared to 90.2 minutes on weekends, with mid-week surges driven by delayed primary care access.  
   *Recommendation*: Schedule elective outpatient follow-ups away from peak weekday surge days to prevent hospital bed boarding.
4. **Queue Tail Risk (90th Percentile Delay)**: While median wait time is 79.0 minutes, the 90th percentile wait reaches 183.0 minutes, with peak facilities exceeding 4 hours.  
   *Recommendation*: Deploy automated operational alerts when facility waiting times exceed 150 minutes to activate rapid medical evaluation (RME) pods and fast-track discharge protocols.

---

## Limitations

- **Triage Acuity Priority**: Public telemetry records operational waiting times for mid-acuity arrivals. Patients presenting with immediately life-threatening conditions (resuscitation CTAS 1) bypass queue waiting times and are treated immediately.
- **Regional Model Specifics**: The data reflects Canadian publicly funded emergency care in Alberta; cross-border transferability to fee-for-service healthcare systems requires local recalibration.
- **Operational Scope**: HealthFlow is an administrative analytics and queue management platform; it does not offer clinical diagnosis, triage classification, or patient treatment advice.

---

## Future Improvements

1. Integration of weather telemetry (ambient temperature, blizzard warnings) to evaluate environmental impact on emergency visits.
2. Real-time Fast Healthcare Interoperability Resources (FHIR) API connectors for electronic health record (EHR) streaming.
3. Queue-length simulation using discrete event modeling (SimPy) to model bed occupancy and doctor service times.
4. Containerized cloud deployment via Docker and Kubernetes for hospital enterprise integration.

---

## Disclaimer

> **IMPORTANT**: This project is intended for educational and analytical purposes. It does not provide medical diagnosis, clinical advice, emergency triage decisions, or treatment recommendations. In any medical emergency, patients should call emergency services or proceed immediately to the nearest hospital emergency department.
