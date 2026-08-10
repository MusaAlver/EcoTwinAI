# EcoTwin AI

[![EcoTwin AI CI](https://github.com/MusaAlver/EcoTwinAI/actions/workflows/ci.yml/badge.svg)](https://github.com/MusaAlver/EcoTwinAI/actions/workflows/ci.yml)

**EcoTwin AI** is an end-to-end intelligent energy monitoring backend for smart buildings.

It combines energy forecasting, adaptive uncertainty estimation, anomaly detection, root-cause attribution, recommendation generation, incident management, REST API serving, automated testing, Docker deployment, and continuous integration.

---

## Engineering Decisions

The main technical decisions, rejected approaches, trade-offs and implementation choices are documented separately:

👉 [Read the Engineering Decisions](docs/engineering-decisions.md)

---

## System Overview

```text
Building Telemetry
        ↓
Feature Engineering
        ↓
30-Minute Energy Forecast
        ↓
Adaptive Uncertainty Threshold
        ↓
Observed Building State
        ↓
Anomaly Detection
        ↓
Root-Cause Attribution
        ↓
Severity & Recommendation
        ↓
Incident Management
        ↓
FastAPI
        ↓
Docker
```

---

## Dataset & Data Preparation

The dataset used in EcoTwin AI was obtained from Kaggle:

**Building 59 Operational Performance Dataset**

https://www.kaggle.com/datasets/gideonkipkorir/building-operational-performance

The dataset contains building operational measurements including whole-building
energy consumption, subsystem-level loads and environmental variables.

The downloaded dataset package also included supporting files such as:

```text
Building59_Vlog.zip
description_table_3year_clean_data.xlsx
metadata_Drayad_Bl9d59.docx
README_Dryad_Bl9d59.txt
```

The raw dataset is not included in this repository. It can be obtained from
the Kaggle source above.

### Data Preparation

The original building data was inspected and processed before model training.
The preparation stage included:

- identifying the building and environmental signals used by the project
- organizing the measurements chronologically
- preparing the time-series structure at 15-minute intervals
- combining relevant HVAC and building-energy variables
- deriving historical power features
- creating time-based features for the forecasting model

Examples of derived variables include:

```text
power_lag_15m
power_lag_60m
power_lag_24h
power_lag_7d

power_delta_15m
power_delta_60m

time_sin
time_cos
dow_sin
dow_cos
is_weekend
```

After preparation, the production forecasting model uses **23 input features**
covering building power, HVAC, MELS, lighting, indoor/outdoor conditions,
historical power behavior and temporal context.

Each prediction uses:

```text
16 timesteps × 23 features
15-minute sampling
≈ 4 hours of historical context
```

The time-series observations were kept in chronological order during model
development rather than randomly shuffling future and past observations.

---
## Energy Forecasting

EcoTwin AI uses a **Gated Residual LSTM** for 30-minute-ahead energy forecasting.

### Model Input

- 16 historical timesteps
- 15-minute sampling interval
- 4 hours of historical context
- 23 engineered features

The final forecast is reconstructed from a persistence baseline and a learned residual correction:

```text
forecast = persistence + 0.42 × predicted_residual
```

### Forecast Performance

| Metric | Result |
|---|---:|
| MAE | **2.776 kW** |
| RMSE | **4.426 kW** |
| Within ±5 kW | **84.63%** |
| Within ±10 kW | **95.58%** |

The **95.58% value is a ±10 kW tolerance hit rate**, not classification accuracy.

---

## Adaptive Uncertainty

Instead of using a fixed anomaly threshold, EcoTwin AI maintains an adaptive conformal threshold.

Current configuration:

- target coverage: **96%**
- rolling calibration window: **30 days**
- delayed error updates
- leakage-aware forecast outcome processing
- clipped anomaly errors during calibration-history updates

This allows the operational threshold to adapt as forecast behavior changes.

---

## Anomaly Detection

Once the real building state becomes available, the system compares it with the earlier forecast.

```text
anomaly_score =
absolute_forecast_error / adaptive_threshold
```

An anomaly is triggered when:

```text
absolute_forecast_error > adaptive_threshold
```

### Controlled Semi-Synthetic Benchmark

| Metric | Result |
|---|---:|
| Precision | **91.92%** |
| Recall | **68.68%** |
| F1 Score | **78.62%** |
| Accuracy | **81.32%** |

These metrics belong specifically to the **controlled semi-synthetic benchmark** and should not be interpreted as real field-labeled anomaly accuracy.

Reference operational alarm rate:

**6.70%**

---

## Root-Cause Attribution

For detected anomalies, EcoTwin AI analyzes subsystem behavior against robust historical expectations.

Current subsystem coverage:

- HVAC North
- HVAC South
- MELS North
- MELS South
- Lighting

Expected subsystem behavior is estimated using local historical context including:

- time of day
- weekday/weekend state
- previous 28 days
- median
- MAD-based robust scale

The root-cause module reports **Attribution Strength** rather than a calibrated probability or confidence score.

### Controlled Root-Cause Benchmark

| Metric | Result |
|---|---:|
| Root-cause accuracy among detected anomalies | **89.75%** |
| End-to-end detection + correct cause | **68.97%** |

These results are based on controlled semi-synthetic subsystem anomaly injections.

---

## Recommendation Engine

EcoTwin AI converts anomaly context into operator-oriented recommendations.

Severity levels:

```text
NORMAL
WARNING
HIGH
CRITICAL
```

Recommendations depend on:

- anomaly score
- energy-consumption direction
- attributed subsystem
- attribution strength

The recommendation layer is currently a **rule-based decision-support engine**, not a separately trained machine-learning model.

---

## Incident Management

Individual alarm points are consolidated into operational incidents.

A new incident is created when:

- consecutive alarms are separated by more than 30 minutes, or
- consumption direction changes

Reference evaluation:

```text
886 alarm points
       ↓
602 incidents
```

This reduces alert volume by approximately **32%**.

Among explained incidents, approximately **84.8%** achieved mean attribution strength of at least 60%.

---

## Architecture

```text
src/
├── forecast.py
├── uncertainty.py
├── anomaly.py
├── root_cause.py
├── recommendation.py
├── incidents.py
├── engine.py
├── runtime.py
└── api.py
```

### Main Components

**forecast.py**  
Loads the trained LSTM model and preprocessing artifacts and produces 30-minute forecasts.

**uncertainty.py**  
Maintains adaptive conformal thresholds.

**anomaly.py**  
Evaluates forecast residuals and determines anomaly state.

**root_cause.py**  
Attributes abnormal consumption to likely subsystem causes.

**recommendation.py**  
Generates severity levels and operational recommendations.

**incidents.py**  
Groups related alarms into higher-level incidents.

**engine.py**  
Connects all intelligence components into the complete EcoTwin pipeline.

**runtime.py**  
Loads calibration artifacts and initializes the engine from disk.

**api.py**  
Exposes EcoTwin AI through FastAPI.

---

## REST API

Available endpoints:

```text
GET  /health
GET  /status
POST /forecast
POST /outcome
GET  /incidents
```

Interactive FastAPI documentation:

```text
http://localhost:8000/docs
```

---

## Example Operational Flow

### 1. Create Forecast

```text
POST /forecast
```

The API returns information such as:

- forecast value
- persistence baseline
- predicted residual
- adaptive threshold
- forecast interval
- forecast ID
- expected outcome time

### 2. Submit Actual Outcome

```text
POST /outcome
```

The system then executes:

```text
Forecast Error
      ↓
Anomaly Detection
      ↓
Root Cause
      ↓
Severity
      ↓
Recommendation
```

---

## Docker

Build the image:

```bash
docker build -t ecotwin-ai:1.0.0 .
```

Run the API:

```bash
docker run -d \
  --name ecotwin-api \
  -p 8000:8000 \
  ecotwin-ai:1.0.0
```

Health check:

```text
http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "service": "EcoTwin AI",
  "version": "1.0.0",
  "engine_loaded": true,
  "initialized": true
}
```

The complete forecast → anomaly → root-cause → recommendation → incident pipeline has also been verified inside the Docker container.

---

## Automated Testing

The project contains automated tests covering:

- forecasting
- uncertainty handling
- anomaly detection
- root-cause attribution
- recommendation generation
- incident management
- integrated engine behavior
- FastAPI endpoints

Current test suite:

```text
36 passed
```

Run locally:

```bash
python -m pytest tests/ -q
```

---

## Continuous Integration

GitHub Actions automatically runs on pushes and pull requests.

```text
Checkout Repository
        ↓
Set Up Python
        ↓
Install Dependencies
        ↓
Run Test Suite
        ↓
Build Docker Image
```

The CI workflow has been successfully verified on GitHub-hosted infrastructure.

---

## Reproducibility

Production artifacts under `models/` include:

- trained Keras model
- input scaler
- residual scaler
- feature definitions
- forecast configuration
- anomaly configuration
- root-cause configuration
- incident configuration
- runtime calibration artifacts
- model integrity manifest

A smoke-test fixture is included to verify that a freshly loaded model reproduces the expected forecast.

---

## Repository Structure

```text
EcoTwinAI/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── models/
├── reports/
├── src/
├── tests/
│
├── 01_data_exploration.ipynb
├── 02_ecotwin_final_pipeline.ipynb
│
├── Dockerfile
├── requirements-api.txt
├── requirements.txt
└── README.md
```

Raw/local datasets are intentionally excluded from Git tracking.

---

## Technology Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn
- FastAPI
- Pydantic
- Uvicorn
- Pytest
- Docker
- GitHub Actions

---

## Current Limitations

EcoTwin AI is currently an engineering and research prototype rather than a field-validated commercial building-management platform.

Important limitations include:

- evaluation currently focuses on one primary building dataset
- the reference dataset does not provide field-labeled anomaly ground truth
- anomaly and root-cause benchmark metrics use controlled semi-synthetic injections
- root-cause analysis currently covers five primary energy subsystems
- the development test period has been inspected during model development
- external unseen-building validation has not yet been completed

---

## Future Work

Planned extensions include:

- external validation on unseen buildings
- multi-building evaluation
- richer digital-twin context
- persistent incident storage
- live operational dashboard
- monitoring and observability
- cloud deployment
- expanded root-cause categories

---

## Goal

EcoTwin AI explores how forecasting and decision-support components can be combined into a reproducible smart-building intelligence pipeline.

The goal is not only to predict future energy consumption, but also to transform building telemetry into **detectable, explainable, and actionable operational information**.

---

## Author

**Muhammed Musa Alver**

GitHub: [MusaAlver](https://github.com/MusaAlver)
