# ⚠️ MODELING RISKS & LIMITATIONS — e0206 Cafeteria AHU

## Executive Summary

This document identifies critical risks and limitations for the N-BEATS forecasting model deployed on device e0206 (Cafeteria AHU). These risks stem from fundamental signal characteristics, data constraints, and modeling assumptions that affect the reliability and business value of the forecasting system.

**Primary Recommendation:** Treat e0206 as a **pipeline validation device**, not a production forecasting target. The low-variance, near-stationary signal makes advanced forecasting methods offer minimal improvement over simple persistence baselines.

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Device** | e0206 (Cafeteria AHU) |
| **Location** | Cafeteria |
| **Assessment Date** | 2026-02-13 |
| **Risk Level** | LOW-MEDIUM (acceptable for baseline experiment) |
| **Production Ready** | ❌ NO (persistence baseline sufficient) |

---

## Risk Category 1: Signal Characteristics

### Risk 1.1 — Low Variance Signal ⚠️ HIGH IMPACT

**Description:**  
Power consumption range is extremely narrow (2.77 - 4.94 kW), with coefficient of variation of only 15.4%. This indicates a near-stationary signal with minimal dynamic behavior.

**Consequence:**
- Model may simply learn to predict the mean or replicate persistence
- Improvement margin is inherently small (±0.5 kW typical variation)
- Complex models cannot demonstrate their value on such stable signals
- Forecasting adds minimal business value when actual consumption is predictable

**Metrics:**
- Power range: 2.17 kW (60% of mean)
- Standard deviation: 0.55 kW
- CV: 15.4% (VERY LOW — stable signals are >30%)
- Dynamic range: 60.4% (moderate relative to mean)

**Mitigation:**
- Set realistic performance expectations (10-20% improvement over persistence is good)
- Use this device only for architecture validation, not forecasting value demonstration
- Consider deploying persistence baseline in production instead of N-BEATS

**When NOT to Deploy:**
- When business value requires >20% forecast improvement
- When variance is needed to demonstrate model capability
- When stakeholders expect "AI magic" to significantly outperform simple heuristics

---

### Risk 1.2 — Weak Seasonality ⚠️ MEDIUM IMPACT

**Description:**  
Diurnal (hourly) and weekly patterns are barely detectable, with <3% variation between time periods. No strong occupancy-driven or schedule-driven load changes.

**Consequence:**
- Seasonal decomposition (trend/seasonal stacks) provides no value
- Time-of-day features contribute minimally to predictions
- Model cannot learn meaningful temporal patterns beyond short-term autocorrelation
- Advanced architectures (seasonal N-BEATS, Prophet) are overkill

**Evidence:**
- Night (0-6h): 3.54 kW
- Day (7-18h): 3.60 kW
- Evening (19-23h): 3.64 kW
- **Variation: <3% across periods**

**Mitigation:**
- Use only generic N-BEATS blocks (no seasonal/trend decomposition)
- Keep architecture lean (3 blocks maximum)
- Focus on short-term autocorrelation (lag-1 to lag-4)
- Do not add seasonal/Fourier features

**When NOT to Deploy:**
- When expecting strong daily/weekly patterns
- When occupancy schedules should drive predictions
- When HVAC setpoints change by time of day

---

### Risk 1.3 — Strong Persistence Effect ⚠️ HIGH IMPACT

**Description:**  
Lag-1 autocorrelation is 0.751 (very high), meaning the best predictor of kw(t) is simply kw(t-1). This creates an extremely strong naive baseline that's hard to beat.

**Consequence:**
- Persistence baseline (prediction = last value) is already strong (MAE ≈ 0.268 kW, 7.5% error)
- N-BEATS must beat this by ≥10% to justify deployment complexity
- If model only learns persistence, it adds no value
- Business case for ML is weak when simple rules work well

**Baseline Performance:**
- **Persistence:** MAE = 0.268 kW (7.53% MAPE)
- **Rolling Mean (4 steps):** MAE = 0.289 kW (worse than persistence)
- **Required N-BEATS performance:** MAE < 0.240 kW (10% improvement minimum)

**Mitigation:**
- Compute and report baseline performance in all evaluations
- Define clear improvement thresholds before declaring success
- If N-BEATS MAE ≥ 0.240 kW, recommend persistence baseline instead
- Measure feature importance to ensure model isn't just learning lag-1

**When NOT to Deploy:**
- When model performance is within 5% of persistence baseline
- When stakeholders expect significant accuracy gains
- When operational changes require advanced pattern recognition

---

## Risk Category 2: Data Limitations

### Risk 2.1 — Small Dataset Size ⚠️ MEDIUM IMPACT

**Description:**  
Only 15 days of data available (~1,357 clean samples after removing flatlines). This is insufficient for robust deep learning model training and generalization.

**Consequence:**
- High risk of overfitting to training data
- Limited ability to capture rare events or edge cases
- Validation/test splits are small (test set ≈ 165 samples)
- Difficult to assess seasonal effects (need months/years)
- Model may not generalize to future operational changes

**Dataset Breakdown:**
- Total expected: 1,440 intervals (15 days × 96 intervals/day)
- After cleaning: ~1,357 intervals (94% retention)
- After windowing: ~1,261 samples (96-step offset)
- **Train:** ~845 samples (67%)
- **Validation:** ~251 samples (20%)
- **Test:** ~165 samples (13%)

**Mitigation:**
- Use strict time-ordered splits (no shuffling) to prevent leakage
- Apply light regularization (dropout 0.1) to reduce overfitting
- Monitor train vs validation loss closely
- Use early stopping (patience = 10 epochs)
- Keep model architecture lean (3 blocks, 128 units)
- Collect more data before production deployment (recommend 3+ months)

**When NOT to Deploy:**
- When validation loss >> training loss (overfitting detected)
- When test performance degrades significantly from validation
- Before collecting at least 1 month of additional data for validation

---

### Risk 2.2 — Flatline Sensitivity ⚠️ HIGH IMPACT

**Description:**  
Sensor failure produced a 21-hour constant power period (83 consecutive flatline intervals at 3.65 kW). This represents 5.8% of the raw dataset and indicates sensor reliability issues.

**Consequence:**
- Model trained on flatline data learns incorrect patterns
- Production deployment requires real-time flatline detection
- If flatlines occur in production, model will produce invalid forecasts
- Data quality monitoring is critical before trusting predictions
- Historical flatline periods must be excluded from training

**Flatline Detection Rules Applied:**
- Threshold: 8+ consecutive intervals with zero change
- Affected intervals: 83 (5.8% of raw data)
- Removed from gold dataset: ✓ (all 83 intervals excluded)

**Mitigation:**
- **In Training:** All flatline periods have been removed from gold dataset
- **In Production:** Implement real-time flatline detection
  - Monitor consecutive zero-change intervals
  - Flag forecasts as "unreliable" during flatline periods
  - Alert operations team to sensor issues
  - Do not retrain on data containing flatlines

**Production Monitoring Requirements:**
```python
# Pseudo-code for production flatline detection
consecutive_zeros = 0
for current_reading in stream:
    if abs(current_reading - previous_reading) < 1e-6:
        consecutive_zeros += 1
        if consecutive_zeros >= 8:
            flag_as_flatline()
            mark_forecast_unreliable()
    else:
        consecutive_zeros = 0
```

**When NOT to Deploy:**
- Without real-time data quality monitoring
- Without sensor health alerting
- If >1% of recent data shows flatlines
- Before validating sensor calibration

---

### Risk 2.3 — No Regime Changes Observed ⚠️ LOW IMPACT

**Description:**  
Device operates continuously 24/7 with no shutdown periods, maintenance windows, or operational mode changes observed in the 15-day sample.

**Consequence:**
- Model has never seen startup/shutdown transients
- Cannot predict behavior during maintenance or failures
- Assumes continuous operation in perpetuity
- May fail catastrophically if device is turned off/on
- Generalization to non-standard operating conditions unknown

**Mitigation:**
- Document assumption: "Model valid only during normal continuous operation"
- If device shuts down, forecasts become invalid
- Retrain model if operational patterns change significantly
- Do not use for capacity planning involving startup/shutdown scenarios

**When NOT to Deploy:**
- During planned maintenance windows
- After significant HVAC system changes
- For capacity scenarios involving off-hours shutdown
- Without operations team awareness of continuous operation assumption

---

## Risk Category 3: Modeling Limitations

### Risk 3.1 — Architecture Overkill for Signal Complexity ⚠️ MEDIUM IMPACT

**Description:**  
N-BEATS with 3 blocks and 128 hidden units (~150K parameters) is potentially over-parameterized for a signal with CV = 15.4% and weak seasonality.

**Consequence:**
- Risk of overfitting on small, stable dataset
- Longer training time than necessary
- Model may learn noise rather than patterns
- Simpler models (ARIMA, persistence) may perform equally well
- Deployment complexity not justified by performance gain

**Model Complexity:**
- **Total parameters:** ~150,000
- **Data samples:** ~1,261
- **Parameter-to-sample ratio:** 119:1 (very high — risk of overfitting)
- **Recommended ratio:** <10:1 for robust generalization

**Mitigation:**
- Use lean architecture (3 blocks only, no seasonal stack)
- Apply dropout (0.1) and early stopping
- Compare against simpler baselines (ARIMA, Exponential Smoothing)
- Consider reducing to 2 blocks or 64 hidden units if overfitting detected
- Monitor validation vs training loss gap

**When NOT to Deploy:**
- If validation loss > training loss by >20%
- If simpler models (ARIMA, persistence) perform within 5% of N-BEATS
- Before experimenting with lighter architectures

---

### Risk 3.2 — Feature Leakage Risk ⚠️ MEDIUM IMPACT

**Description:**  
Apparent power (`apparent_power_total`) has 0.991 correlation with target (`power_total`). This near-perfect correlation indicates potential feature leakage or redundancy.

**Consequence:**
- Model may rely on leaked features rather than temporal patterns
- Feature importance analysis will be misleading
- Generalization to scenarios where apparent power isn't available will fail
- Model learns associations, not causal relationships

**High-Correlation Features:**
| Feature | Correlation | Risk Level |
|---------|-------------|------------|
| `apparent_power_total` | 0.991 | **VERY HIGH** (likely redundant) |
| `power_factor_avg` | 0.957 | **HIGH** |
| `current_l1` | 0.922 | **HIGH** |
| `power_l1` | 0.932 | **HIGH** |

**Mitigation:**
- **Option 1:** Exclude `apparent_power_total` from features entirely
- **Option 2:** Monitor feature importance; if apparent power dominates, exclude it
- Use only electrical measurements that would be available in production
- Validate that model learns temporal patterns, not feature shortcuts
- Perform ablation study: train with/without high-correlation features

**When NOT to Deploy:**
- If apparent power feature has >50% importance
- Without feature importance analysis
- Before confirming features available in production inference

---

### Risk 3.3 — Baseline Ceiling Effect ⚠️ HIGH IMPACT

**Description:**  
The strong persistence baseline (MAE = 0.268 kW) creates a performance ceiling that's difficult to meaningfully surpass. Even 20% improvement only reduces MAE to 0.214 kW — a marginal operational difference.

**Consequence:**
- N-BEATS may not beat persistence by ≥10% (minimum threshold)
- Business value of forecasting is questionable
- Operational decisions unlikely to change with 0.05 kW better forecasts
- Investment in ML infrastructure may not be justified

**Performance Thresholds:**
| Outcome | MAE Threshold | Improvement | Decision |
|---------|---------------|-------------|----------|
| **Excellent** | <0.190 kW | >30% | ✓ Deploy N-BEATS |
| **Good** | 0.190 - 0.215 kW | 20-30% | Consider N-BEATS |
| **Acceptable** | 0.215 - 0.240 kW | 10-20% | Marginal value |
| **Insufficient** | >0.240 kW | <10% | ❌ Use persistence |

**Mitigation:**
- Define success criteria before training (not after)
- If N-BEATS MAE ≥ 0.240 kW, recommend persistence baseline
- Calculate business impact of forecast improvement
  - Does 0.05 kW better accuracy change operational decisions?
  - What is cost of ML infrastructure vs. simple rules?
- Be transparent with stakeholders about diminishing returns

**When NOT to Deploy:**
- If improvement over persistence is <10%
- Without clear business case for marginal accuracy gains
- When persistence baseline meets operational requirements

---

## Risk Category 4: Operational Risks

### Risk 4.1 — Misleading Accuracy Metrics ⚠️ MEDIUM IMPACT

**Description:**  
Low-variance signals naturally achieve high R² and low MAPE, which can create false confidence in model quality. A trivial model predicting the mean gets R² ≈ 0.9 on this signal.

**Consequence:**
- Stakeholders may overestimate model value based on high R²
- MAPE of 5% sounds impressive but represents minimal absolute improvement
- Model performance looks better than it actually is
- Comparison to other devices/use cases is misleading

**Example Misleading Metrics:**
- **N-BEATS R²:** 0.92 ← Sounds great
- **Persistence R²:** 0.90 ← Only 2% worse
- **Actual MAE difference:** 0.03 kW ← Minimal operational impact

**Mitigation:**
- **Primary metric:** MAE (kW) — absolute error in power units
- **Secondary metric:** % improvement over persistence baseline
- **Avoid emphasizing:** R², MAPE (misleading on low-variance signals)
- Report baseline performance prominently alongside model results
- Contextualize errors: "0.268 kW error is 7.5% of mean 3.6 kW"

**When NOT to Deploy:**
- When stakeholders focus only on R² without understanding baseline
- Without clear communication of absolute vs. relative performance
- Before educating decision-makers on low-variance signal characteristics

---

### Risk 4.2 — Generalization to New Operating Conditions ⚠️ HIGH IMPACT

**Description:**  
Model trained on 15 days of "normal" operation may not generalize to:
- Seasonal changes (winter vs. summer)
- Equipment aging
- Maintenance-induced behavior shifts
- Occupancy pattern changes (e.g., pandemic, schedule changes)

**Consequence:**
- Model performance may degrade over time
- Forecasts become unreliable after operational changes
- Retraining required but data pipeline may not detect this need
- Silent failures (bad predictions without alerts)

**Mitigation:**
- Monitor forecast error in production continuously
- Set error threshold alerts (e.g., MAE > 0.35 kW for 24 hours)
- Retrain model quarterly or when error threshold exceeded
- Collect data across seasons before claiming robust performance
- Document operational conditions during training

**When NOT to Deploy:**
- Without production monitoring and alerting
- Before collecting data across multiple seasons
- Without automatic retraining pipeline
- If operational conditions change frequently

---

## Risk Category 5: Business & ROI Risks

### Risk 5.1 — Unclear Business Value ⚠️ HIGH IMPACT

**Description:**  
For a continuously running, low-variance cafeteria AHU, the business value of 15-minute-ahead power forecasts is unclear. What operational decision would change based on these forecasts?

**Consequence:**
- ML infrastructure costs (compute, maintenance, monitoring) may exceed value delivered
- Stakeholders disappointed when "AI forecasting" doesn't translate to savings
- Resource allocation to low-impact use case
- Opportunity cost of not working on higher-value forecasting targets

**Questions to Answer Before Production:**
1. What decision will change based on 15-min ahead forecasts?
2. What is the cost of forecast errors?
3. Is demand response or load shedding applicable to this device?
4. Can forecasts reduce energy costs or improve reliability?
5. What is the ROI of ML system vs. simple persistence?

**Mitigation:**
- Define clear business use case before deployment
- Calculate ROI: (value of forecast improvement) - (ML system costs)
- Prioritize devices where forecasting enables actionable decisions
- Consider e0206 as validation device only, not production target

**When NOT to Deploy:**
- Without clear business use case
- When stakeholders can't articulate operational value
- Before cost-benefit analysis
- If persistence baseline meets all business requirements

---

## Strategic Recommendations

### ✅ DO Deploy N-BEATS on e0206 For:

1. **Pipeline Validation**
   - Prove end-to-end data extraction → training → evaluation works
   - Validate architecture implementation correctness
   - Establish baseline for more complex devices

2. **Architecture Benchmarking**
   - Test N-BEATS on well-behaved, clean data
   - Understand performance on low-complexity signals
   - Document lessons learned for future devices

3. **Team Learning**
   - Build institutional knowledge of time series forecasting
   - Develop monitoring and evaluation frameworks
   - Practice model lifecycle management

### ❌ DO NOT Deploy N-BEATS on e0206 For:

1. **Production Forecasting**
   - Persistence baseline (MAE = 0.268 kW) is sufficient
   - Marginal improvement doesn't justify ML complexity
   - Business value unclear for 15-min cafeteria AHU forecasts

2. **Demonstrating ML Value**
   - Low-variance signal doesn't showcase model capabilities
   - Stakeholders won't be impressed by 10% improvement
   - Better devices exist to prove forecasting value

3. **Long-Term Deployment**
   - Small dataset limits generalization confidence
   - Continuous operation assumption fragile
   - Retraining burden high for minimal benefit

---

## Decision Framework: When to Deploy on e0206

Use this flowchart to decide if production deployment is warranted:

```
┌─────────────────────────────────────┐
│ Did N-BEATS beat persistence by ≥10%│
└──────────────┬──────────────────────┘
               │
         ┌─────┴─────┐
         │    NO     │──────► ❌ DO NOT DEPLOY
         └───────────┘         Use persistence baseline
               │
         ┌─────┴─────┐
         │   YES     │
         └─────┬─────┘
               │
    ┌──────────▼────────────┐
    │ Is there a clear      │
    │ business use case?    │
    └──────────┬────────────┘
               │
         ┌─────┴─────┐
         │    NO     │──────► ⚠️  RECONSIDER
         └───────────┘         Define use case first
               │
         ┌─────┴─────┐
         │   YES     │
         └─────┬─────┘
               │
    ┌──────────▼────────────┐
    │ Is production          │
    │ monitoring in place?   │
    └──────────┬────────────┘
               │
         ┌─────┴─────┐
         │    NO     │──────► ⚠️  NOT READY
         └───────────┘         Build monitoring first
               │
         ┌─────┴─────┐
         │   YES     │
         └─────┬─────┘
               │
          ✅ DEPLOY
      (with caution)
```

---

## Monitoring Requirements (If Deployed)

If you proceed with production deployment despite risks, implement these monitoring safeguards:

### 1. Data Quality Monitoring
- Real-time flatline detection (8+ consecutive zero-change intervals)
- Range validation (2.5 - 5.5 kW acceptable bounds)
- Missing data alerts
- Sensor health checks

### 2. Model Performance Monitoring
- Track MAE on rolling 24-hour window
- Alert if MAE > 0.35 kW (30% worse than baseline)
- Compare to persistence baseline weekly
- Log feature importance to detect drift

### 3. Operational Alerts
- Forecast error > 0.5 kW for single timestep
- 3+ consecutive errors > 0.4 kW
- Validation loss increase > 20% from training baseline
- Data distribution shift detection

### 4. Retraining Triggers
- Quarterly retraining minimum
- Error threshold exceeded for 48 hours
- Operational changes reported by facilities team
- Sensor maintenance or calibration events

---

## Conclusion

**e0206 is an excellent pipeline validation device but a poor production forecasting target.**

The combination of:
- Low signal variance (CV = 15.4%)
- Weak seasonality (<3% variation)
- Strong persistence (lag-1 autocorr = 0.751)
- Small dataset (15 days)
- Unclear business value

...makes advanced forecasting methods offer minimal improvement over simple baselines.

**Recommended Path Forward:**

1. ✅ **Complete N-BEATS training** as planned for pipeline validation
2. ✅ **Document performance** vs. persistence baseline
3. ✅ **Extract learnings** about architecture and data requirements
4. ⚠️  **Do not deploy to production** unless N-BEATS beats persistence by ≥20% AND clear business case exists
5. ✅ **Move to higher-complexity devices** (variable load, strong seasonality, shutdown periods)
6. ✅ **Use persistence baseline** for e0206 production forecasting

**Success Definition:**  
If N-BEATS achieves MAE < 0.215 kW (20% improvement) → Architecture validated ✓  
If N-BEATS achieves MAE ≥ 0.240 kW (<10% improvement) → Persistence sufficient ✓  

**Both outcomes are valuable learnings.** This is a validation experiment, not a production deployment.

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-13 | Data Science Team | Initial risk assessment |

**Next Review:** After first model training run  
**Status:** APPROVED FOR BASELINE EXPERIMENT  
**Production Deployment:** ❌ NOT RECOMMENDED
