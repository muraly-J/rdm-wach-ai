# Data Feasibility Report – AHU Metrics

## 1. Overview
This report summarizes the feasibility of applying machine learning (ML) models to various Air Handling Unit (AHU) performance metrics. The analysis considers:

- Anomaly coverage (percentage of unhealthy hours globally)
- Percentage of AHUs affected
- Stationarity of metrics (suitability for ML modeling)

The objective is to decide **which metrics are ready for ML** and which need **more operational data**.

---

## 2. Data Coverage & Findings
- **Total AHUs analyzed:** 154  
- **Metrics evaluated:** Energy, Power Factor (PF), Phase Imbalance, THD, Overload  
- **Global anomaly percentages:**  
  - Energy: 49.77%  
  - PF: 28.66%  
  - Phase Imbalance: 30.43%  
  - THD: 29.68%  
  - Overload: 6.50%  

- **Stationarity (ML-feasible hours):**
  - Energy: 83.77%  
  - PF: 4.55%  
  - Phase Imbalance: 11.04%  
  - THD: 38.96%  
  - Overload: 10.39%  

> **Interpretation:** Metrics with low stationarity (<50%) are not currently suitable for ML.

---

## 3. Go / No-Go Assessment

## Go / No-Go Assessment per Component

| Metric / Component       | Anomaly % | AHUs Affected % | Stationary % | Decision | Reason |
|--------------------------|-----------|----------------|--------------|----------|--------|
| Energy Anomaly           | 49.77     | 99.35          | 83.77        | GO       | Stationary % high, anomaly coverage sufficient |
| PF Degradation           | 28.66     | 99.35          | 4.55         | NO-GO    | Stationary % too low |
| Phase Imbalance          | 30.43     | 99.35          | 11.04        | NO-GO    | Stationary % too low |
| THD Drift                | 29.68     | 100.00         | 38.96        | NO-GO    | Stationary % below threshold |
| Overload                 | 6.50      | 73.38          | 10.39        | NO-GO    | Stationary % too low, anomaly 

**Decision Criteria:**  
- Anomaly % ≥ 5%  
- AHUs Affected % ≥ 20%  
- Stationary % ≥ 50% → GO for ML  

---

## 4. Recommendations
- **ML-ready now:**  
  - **Energy Anomaly** (sufficient anomaly coverage, AHU coverage, and stationarity)

- **Need more data / preprocessing:**  
  - **PF Degradation** – Stationarity too low  
  - **Phase Imbalance** – Stationarity too low  
  - **THD Drift** – Stationarity below 50%  
  - **Overload** – Low anomaly coverage and stationarity  

> **Action Plan:**  
> Proceed with ML model development for Energy prediction. For other metrics, collect an additional 3–6 months of rule-based logging to improve stationarity and anomaly coverage before applying ML.

---

## 5. Risks & Notes
- Low stationarity in PF, THD, Phase Imbalance, and Overload could lead to poor ML model performance.  
- Recommendations are based on **current AHU operational data**; metrics can be re-evaluated as more data accumulates.


“We evaluated 5 AHU performance metrics for ML feasibility using our GO/NO-GO framework: anomaly coverage, AHUs affected, and stationarity.
Results:

Energy Anomaly → GO, ready for ML

PF Degradation, Phase Imbalance, THD Drift, Overload → NO-GO, stationarity too low
Recommendation: start ML modeling for energy usage predictions. For the other metrics, continue logging data for the next 3–6 months before ML application. This ensures model reliability and prevents inaccurate predictions.”