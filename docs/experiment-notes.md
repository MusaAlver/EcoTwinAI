# EcoTwin AI — Experiment Notes

This document summarizes selected experiments that influenced the final EcoTwin AI design.

These notes are not intended to represent every experiment performed during development. They document several comparisons that directly affected the production pipeline.

---

## 1. Forecast Horizon Comparison

Two practical forecasting horizons were considered:

- 30-minute forecasting
- 60-minute forecasting

A simple persistence forecast was kept as the baseline.

### Observation

For the 30-minute horizon, the residual LSTM provided enough value to justify using the learned model.

For the 60-minute horizon, the machine-learning approach did not improve MAE enough to justify the additional complexity.

### Result

```text
30-minute horizon → Gated Residual LSTM
60-minute horizon → Persistence
```

The decision was based on evaluation performance rather than using machine learning for every forecasting horizon.

---

## 2. Residual Forecasting

Direct load prediction was not the only strategy considered.

The final 30-minute model predicts a residual correction relative to the current persistence estimate.

```text
persistence
    +
learned correction
    ↓
final forecast
```

The final correction gate was:

```text
lambda = 0.42
```

Final formulation:

```text
forecast = persistence + 0.42 × predicted_residual
```

### Final 30-Minute Evaluation

| Metric | Result |
|---|---:|
| MAE | 2.776 kW |
| RMSE | 4.426 kW |
| Within ±5 kW | 84.63% |
| Within ±10 kW | 95.58% |

The ±10 kW result is treated as a tolerance hit rate rather than classification accuracy.

---

## 3. IsolationForest Anomaly Experiment

IsolationForest was evaluated as an anomaly-detection option.

### Problem Observed

Later portions of the time series differed from the distribution represented by earlier data.

As the distribution shifted, IsolationForest started marking an unrealistic amount of the later observations as anomalous.

This behavior suggested that concept drift would make the detector noisy in this setting.

### Decision

IsolationForest was removed from the final operational pipeline.

The final system instead defines abnormal behavior through forecast residuals and an adaptive uncertainty threshold.

---

## 4. Adaptive Threshold Selection

Several uncertainty coverage levels were compared on validation data.

The selected operating point was:

```text
Coverage = 96%
```

The selection rule prioritized keeping the validation false-alarm level within the chosen operating constraint while obtaining the strongest controlled anomaly-detection performance.

The final adaptive system uses:

```text
30-day rolling calibration history
delayed outcome updates
finite-sample conformal quantile
clipped anomaly-history updates
```

---

## 5. Controlled Anomaly Evaluation

The reference dataset does not provide reliable field-labeled anomaly ground truth.

To evaluate known abnormal conditions, controlled semi-synthetic deviations were introduced.

At the selected 96% coverage:

| Metric | Result |
|---|---:|
| Precision | 91.92% |
| Recall | 68.68% |
| F1 | 78.62% |
| Accuracy | 81.32% |

These metrics are therefore reported as a **controlled semi-synthetic benchmark**.

They are not presented as real-world anomaly accuracy.

---

## 6. Root-Cause Evaluation

Controlled subsystem deviations were also used to evaluate whether EcoTwin could identify the subsystem responsible for a detected anomaly.

Results:

| Evaluation | Result |
|---|---:|
| Root-cause accuracy among detected injected anomalies | 89.75% |
| End-to-end detected + correct cause | 68.97% |

The current controlled benchmark primarily represents positive subsystem deviations.

This limitation is kept explicit in the project documentation.

---

## 7. Alert vs Incident Representation

The operational evaluation produced:

```text
886 alarm points
```

Treating each timestamp as a separate operational event created unnecessary repetition.

An incident-grouping layer was therefore introduced.

Result:

```text
886 alarms
   ↓
602 incidents
```

Approximately 32% fewer operational events were presented after grouping.

---

## 8. Reproducibility Check

The forecasting pipeline was not considered complete when it only worked inside the development notebook.

The production artifacts were exported and loaded again in a fresh runtime.

A saved smoke-test fixture was then used to verify that the reloaded model reproduced the expected forecast.

The numerical difference was approximately:

```text
1e-6 kW
```

This confirmed that the production forecast was reproducible independently of the original notebook state.

---

## Summary

The final EcoTwin architecture was shaped by experimental results rather than by selecting the most complex method available.

Examples include:

```text
ML beats useful baseline → use ML
ML does not beat baseline → keep baseline

Detector becomes unstable under drift → reject it

Fixed anomaly limits are insufficient → use adaptive limits

No field labels available → label benchmark appropriately

Repeated alarms create noise → aggregate into incidents
```

These experiments directly influenced the final production design.
