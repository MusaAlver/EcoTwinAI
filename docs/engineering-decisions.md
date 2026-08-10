# EcoTwin AI — Engineering Decisions

This document records the main technical decisions made during the development of EcoTwin AI.

The purpose is to explain not only **what was implemented**, but also **why specific approaches were selected or rejected**.

---

## Decision 01 — Use a residual LSTM for the 30-minute forecast

### Problem

A simple persistence forecast is already a strong baseline for short-horizon building energy prediction.

A forecasting model therefore needs to improve on the baseline rather than simply learn the absolute load from scratch.

### Approach

The final 30-minute model was designed as a residual forecasting system.

Instead of directly predicting the complete future load, the LSTM predicts a correction relative to the persistence baseline.

Final formulation:

```text
forecast = persistence + 0.42 × predicted_residual
```

### Final Decision

Use the **Gated Residual LSTM** for the 30-minute forecast.

### Why

The residual formulation keeps the strong persistence behavior while allowing the model to correct systematic deviations.

This also makes the model easier to compare against the baseline.

---

## Decision 02 — Keep persistence for the 60-minute forecast

### Problem

A machine-learning model is not automatically better than a simple baseline.

During evaluation, the 60-minute ML forecast did not provide a meaningful improvement over persistence.

### Final Decision

Use **persistence** as the production strategy for the 60-minute horizon.

### Why

The persistence baseline produced slightly better MAE and required less model complexity.

A more complex model was therefore not justified simply for the sake of using machine learning.

---

## Decision 03 — Reject IsolationForest as the final anomaly detector

### Problem

IsolationForest was considered as an anomaly-detection approach.

However, building-energy behavior changes over time.

When the operational distribution shifted, the model started treating a very large portion of the later data as anomalous.

### Observation

The anomaly rate became unrealistic because the detector was sensitive to concept drift.

This would generate excessive operational noise.

### Final Decision

Do not use IsolationForest as the production anomaly detector.

Instead, detect anomalies using the relationship between:

```text
actual consumption
        ↓
forecast expectation
        ↓
adaptive forecast-error threshold
```

### Why

The forecasting residual provides a more meaningful definition of abnormal behavior:

> Is the building behaving significantly differently from what the forecasting system expected?

---

## Decision 04 — Use adaptive conformal thresholds instead of a fixed threshold

### Problem

A fixed rule such as:

```text
error > 5 kW
```

does not account for periods where the forecasting model naturally becomes more or less uncertain.

### Approach

EcoTwin AI uses an adaptive conformal threshold based on recent forecast errors.

Current configuration:

```text
Target coverage: 96%
Rolling calibration window: 30 days
Forecast horizon: 30 minutes
```

Forecast errors are added only after their outcomes become available.

Extreme anomaly errors are clipped before updating the calibration history so that a large anomaly does not permanently inflate future thresholds.

### Final Decision

Use a **rolling adaptive conformal threshold** for operational anomaly detection.

### Why

The anomaly boundary should adapt to recent forecasting behavior rather than remain permanently fixed.

---

## Decision 05 — Keep anomaly benchmark claims scientifically limited

### Problem

The reference building dataset does not provide reliable field-labeled anomaly ground truth.

Without ground-truth anomaly labels, a true real-world anomaly accuracy cannot be calculated.

### Approach

Controlled anomalies were injected to evaluate whether the system can detect known deviations.

### Final Decision

Report these results explicitly as a:

**controlled semi-synthetic benchmark**

and not as real-world field anomaly accuracy.

Current benchmark results:

| Metric | Result |
|---|---:|
| Precision | 91.92% |
| Recall | 68.68% |
| F1 | 78.62% |
| Accuracy | 81.32% |

### Why

The benchmark is useful for controlled validation, but its limitations should remain visible.

---

## Decision 06 — Use "Attribution Strength" instead of "confidence"

### Problem

The root-cause module ranks subsystem deviations using robust historical statistics.

Its final score is not a calibrated probability.

Calling the result:

```text
98% confidence
```

would therefore overstate what the number represents.

### Final Decision

Use the term:

**Attribution Strength**

### Why

The value describes how strongly the available subsystem evidence supports one cause relative to the others.

It should not be interpreted as a probabilistic confidence estimate.

---

## Decision 07 — Convert alarm points into incidents

### Problem

A persistent energy problem can create several consecutive 15-minute alarm points.

Treating every point as an independent problem creates alert noise.

Reference evaluation produced:

```text
886 alarm points
```

### Approach

Related alarms are grouped into incidents.

A new incident begins when:

- the gap between alarms exceeds 30 minutes, or
- the energy deviation direction changes.

### Result

```text
886 alarm points
        ↓
602 incidents
```

This reduced the number of operational events by approximately 32%.

### Final Decision

Expose incidents rather than relying only on individual alarm points.

### Why

Operational users usually care about one continuing problem, not every timestamp generated during that problem.

---

## Decision 08 — Separate research code from production modules

### Problem

The first stages of the project were developed interactively in Jupyter notebooks.

This is useful for data exploration and experimentation but becomes difficult to maintain as a production-style system grows.

### Final Decision

Move the main runtime logic into modular Python components:

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

The notebooks remain as experimentation and research records.

### Why

This separation makes the project:

- easier to test
- easier to reuse
- easier to deploy
- easier to review
- less dependent on notebook state

---

## Decision 09 — Test the system after cold loading

### Problem

A machine-learning notebook can appear to work because objects and variables remain in memory.

That does not prove the exported system is reproducible.

### Approach

The final model, scalers, configuration files and calibration artifacts were exported to disk.

The runtime was then initialized again from those saved artifacts.

A smoke-test fixture verifies that the freshly loaded forecasting model reproduces the expected prediction.

### Final Decision

Treat cold-load reproducibility as part of the production contract.

---

## Decision 10 — Containerize the complete API

### Problem

A system that only runs inside one local Python environment is difficult to reproduce elsewhere.

### Final Decision

Package the EcoTwin runtime using Docker.

The container includes:

```text
FastAPI
EcoTwin runtime modules
trained model
scalers
configuration
calibration artifacts
```

The containerized system was verified using:

```text
health check
forecast request
anomaly detection
root-cause attribution
recommendation generation
incident retrieval
```

### Why

This provides a reproducible deployment boundary rather than relying on the developer's local environment.

---

## Development Principle

A recurring principle in EcoTwin AI has been:

> Prefer the simplest approach that performs the required task reliably, and avoid presenting experimental results as stronger evidence than they actually provide.

For that reason:

- persistence was retained where it outperformed ML,
- IsolationForest was rejected when drift made it unsuitable,
- semi-synthetic metrics are labeled explicitly,
- root-cause scores are not presented as probabilities,
- test-set limitations are documented,
- and deployment behavior is tested independently from the notebooks.
