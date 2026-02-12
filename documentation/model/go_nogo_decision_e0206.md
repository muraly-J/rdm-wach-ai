# Go/No-Go Modeling Decision — e0206 (Cafeteria AHU)

## Executive Summary

**DECISION: 🟢 GO — WITH STRATEGIC CAVEATS**

**Primary Purpose:** Pipeline validation and baseline establishment  
**Secondary Purpose:** Benchmarking architecture on low-complexity device  
**Not Recommended For:** Production forecasting (persistence baseline sufficient)

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Decision Date** | 2026-02-12 |
| **Device** | e0206 (Cafeteria AHU) |
| **Data Period** | 2026-01-28 to 2026-02-11 (15 days) |
| **Decision Maker** | Data Science Team |
| **Review Date** | After first model training |

---

## 1. Signal Characteristics Assessment

### 1.1 Target Variable Profile

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Range** | 2.77 - 4.94 kW | Narrow (78% span) |
| **Mean** | 3.60 kW | Stable baseline |
| **Std Dev** | 0.55 kW | Low variance |
| **CV** | 15.4% | **VERY LOW** (stable signal) |
| **Dynamic Range** | 60.4% | Moderate relative to mean |

**Assessment:** ⚠️ Low-variance, near-stationary signal

### 1.2 Temporal Behavior

| Pattern Type | Magnitude | % of Mean | Status |
|--------------|-----------|-----------|--------|
| **Hourly variation** | 0.29 kW | 7.9% | WEAK |
| **Daily variation** | 0.50 kW | 13.8% | WEAK-MODERATE |
| **No shutdown periods** | - | - | Continuous operation |

**Time-of-Day Effect:**
- Night (0-6h): 3.54 kW
- Day (7-18h): 3.60 kW  
- Evening (19-23h): 3.64 kW
- **Difference:** <3% between periods

**Assessment:** ⚠️ Minimal diurnal seasonality

### 1.3 Autocorrelation Structure

| Lag | Time Period | Autocorr | Interpretation |
|-----|-------------|----------|----------------|
| 1 | 15 minutes | **0.751** | **Very high** — persistence dominant |
| 4 | 1 hour | 0.616 | High — recent history matters |
| 96 | 24 hours | 0.533 | Moderate — some daily structure |

**Assessment:** ✅ Strong short-term predictability

### 1.4 Change Dynamics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean absolute change (15-min) | 0.27 kW | 7.5% of mean |
| Max change | 1.05 kW | Rare spike |
| 95th percentile change | 0.86 kW | Typical max variation |

**Assessment:** ✅ Smooth, predictable transitions

---

## 2. Data Quality Analysis

### 2.1 Quality Metrics

| Dimension | Status | Details |
|-----------|--------|---------|
| **Missing Data** | ✅ Excellent | 0% NaN (perfect completeness) |
| **Sampling** | ✅ Perfect | 1,440/1,440 expected intervals |
| **Flatlines** | ⚠️ Detected | 83 intervals (5.8%) — will be filtered |
| **Outliers** | ✅ None | All values within expected range |

### 2.2 Expected Gold Dataset

| Metric | Value |
|--------|-------|
| Raw rows | 1,440 |
| Flatline rows (to remove) | 83 |
| **Gold dataset rows** | **~1,357** |
| **Gold percentage** | **94.2%** |

**Assessment:** ✅ Exceptionally clean data

---

## 3. Baseline Performance Analysis

### 3.1 Persistence Model (Naive)

**Model:** `kw_pred(t) = kw(t-1)`

| Metric | Value |
|--------|-------|
| MAE | 0.268 kW |
| RMSE | 0.390 kW |
| MAPE | 7.53% |
| Relative Error | 7.45% of mean |

**Assessment:** ⚠️ **Very strong baseline** — hard to beat significantly

### 3.2 Rolling Mean Model

**Model:** `kw_pred(t) = mean(kw[t-4:t-1])`  (1-hour window)

| Metric | Value |
|--------|-------|
| MAE | 0.289 kW |
| RMSE | 0.371 kW |

**Performance vs Persistence:** -7.7% (WORSE)

**Assessment:** ⚠️ Smoothing hurts performance due to high lag-1 autocorr

### 3.3 Required Model Performance

For N-BEATS to add value, must achieve:

| Threshold | Target | Rationale |
|-----------|--------|-----------|
| **Minimum** | MAE < 0.240 kW | 10% improvement over persistence |
| **Good** | MAE < 0.215 kW | 20% improvement |
| **Excellent** | MAE < 0.190 kW | 30% improvement |

**If model achieves < 10% improvement:** Forecasting not justified for this device.

---

## 4. Modeling Complexity Assessment

### 4.1 Forecasting Difficulty Score

**Formula:** `(σ/μ) × (1 - ρ₁) × (1 - pattern_strength/10)`

**Score:** 0.0328

**Classification:** **VERY EASY**

| Range | Classification | e0206 Status |
|-------|----------------|--------------|
| < 0.05 | Very Easy | ✅ **HERE** |
| 0.05 - 0.15 | Easy | |
| 0.15+ | Moderate-Hard | |

### 4.2 Signal Complexity

| Indicator | Value | Interpretation |
|-----------|-------|----------------|
| Signal-to-Noise Ratio | 6.51 | Good (>5 = strong signal) |
| Entropy (normalized std) | 0.154 | Low (stable) |
| Pattern Strength | 1.41 | Moderate |

**Assessment:** Low-entropy signal with moderate patterns

---

## 5. Feature Correlation Analysis

### 5.1 Correlation with Target (power_total)

| Feature | Correlation | Strength | Use Case |
|---------|-------------|----------|----------|
| `apparent_power_total` | **0.991** | Extreme | Almost redundant with target |
| `power_factor_avg` | **0.957** | Extreme | Highly predictive |
| `current_l1` | **0.922** | Very High | Strong predictor |
| `power_l1` | **0.932** | Very High | Strong predictor |
| `current_l3` | 0.799 | High | Useful |
| `power_l3` | 0.796 | High | Useful |
| `power_l2` | 0.280 | Low | Weak (due to L2 stability) |
| `current_l2` | 0.207 | Low | Weak |

**Assessment:** ✅ Strong feature-target relationships exist

**⚠️ Risk:** Apparent power correlation (0.991) may lead to feature leakage

---

## 6. Strategic Considerations

### 6.1 Why GO?

| Reason | Impact | Priority |
|--------|--------|----------|
| **Pipeline validation** | Proves end-to-end extraction → model → eval | HIGH |
| **Architecture benchmarking** | Tests N-BEATS on well-behaved data | HIGH |
| **Clean data** | No quality issues to debug during model dev | MEDIUM |
| **Strong features** | Clear physical relationships to learn | MEDIUM |
| **Known baseline** | Clear performance target (beat 0.268 MAE) | HIGH |

### 6.2 Why Risky?

| Concern | Impact | Mitigation |
|---------|--------|------------|
| **Low variance signal** | Model may not learn much | Treat as baseline; don't over-invest |
| **Strong persistence** | Naive baseline hard to beat | Set clear performance thresholds |
| **Weak seasonality** | Complex architectures may be overkill | Use lean N-BEATS (3 blocks) |
| **Continuous operation** | No regime changes to learn | Simple architecture sufficient |
| **Small dataset** | 15 days only | Conservative train/val/test split |

### 6.3 What This Device Is NOT

| This is NOT | Because |
|-------------|---------|
| A forecasting challenge | Persistence baseline already strong |
| A complex signal | Low variance, weak seasonality |
| A production use case | No business need to forecast 15 min ahead |
| An anomaly detection case | Stable operation, known quirks |

**This IS:** A pipeline validation experiment and architecture baseline.

---

## 7. Go/No-Go Decision Matrix

### 7.1 Critical Go Criteria

| Criterion | Required | Status |
|-----------|----------|--------|
| Clean data available | Yes | ✅ 94% gold dataset |
| Target variable valid | Yes | ✅ kw well-defined |
| Features meaningful | Yes | ✅ Strong correlations |
| Baseline defined | Yes | ✅ Persistence = 0.268 MAE |
| Clear success metric | Yes | ✅ Beat baseline by 10%+ |

**All critical criteria met:** ✅

### 7.2 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Model doesn't beat baseline | HIGH | Medium | Expected; document findings |
| Overfitting on small dataset | MEDIUM | Medium | Strict time-ordered split |
| Feature leakage (apparent_power) | LOW | High | Monitor; exclude if needed |
| Weak generalization | MEDIUM | Low | Only 1 device anyway |

**Overall Risk Level:** LOW-MEDIUM (acceptable for baseline experiment)

### 7.3 Expected Outcomes

| Scenario | Probability | Action |
|----------|-------------|--------|
| Model beats baseline by >20% | LOW (20%) | ✅ Proceed to more devices |
| Model beats baseline by 10-20% | MEDIUM (40%) | ⚠️ Marginal; evaluate complexity |
| Model beats baseline by <10% | HIGH (40%) | ❌ Persistence sufficient |

---

## 8. Final Decision

### 8.1 Official Verdict

**🟢 GO FOR MODELING**

**With the following conditions:**

1. **Purpose:** Treat as pipeline validation, not production deployment
2. **Architecture:** Use lean N-BEATS (3 generic blocks, no seasonal stack)
3. **Benchmark:** Must beat persistence (MAE 0.268) by ≥10% to justify complexity
4. **Timeline:** Fast iteration (1-2 days max for initial results)
5. **Next step:** If successful, move to higher-complexity devices

### 8.2 Success Definition

**Model is considered successful if:**

- MAE < 0.240 kW (10% improvement)
- RMSE < 0.350 kW
- No evidence of overfitting (val ≈ test performance)
- Training completes without errors

**Model is considered excellent if:**

- MAE < 0.215 kW (20% improvement)
- Captures temporal patterns persistence misses

**If model fails:**

- Document that persistence is sufficient for this device type
- Use findings to inform device selection criteria for future modeling
- Still counts as successful pipeline validation

### 8.3 Resource Allocation

| Resource | Allocation | Justification |
|----------|------------|---------------|
| **Dev time** | 2-3 days | Quick baseline experiment |
| **Compute** | Minimal (CPU sufficient) | Small dataset, simple architecture |
| **Priority** | Medium | Baseline, not critical path |

---

## 9. Recommendations

### 9.1 Immediate Actions

1. ✅ Proceed with data extraction using validated spec
2. ✅ Implement lean N-BEATS architecture (see modeling spec)
3. ✅ Set up baseline comparison pipeline
4. ✅ Use strict time-ordered train/val/test split

### 9.2 Monitoring During Training

**Watch for:**

- Val loss > train loss (overfitting)
- Model performance not improving over persistence by epoch 10
- Feature importance: if only lag-1 matters, model is learning persistence

### 9.3 Post-Modeling Actions

**If successful (beats baseline by 10%+):**
- Document learned patterns
- Extract feature importance
- Identify which temporal features helped
- Move to next device (higher complexity)

**If unsuccessful (<10% improvement):**
- Document that continuous, low-variance devices don't need forecasting
- Recommend persistence model for this device class
- Adjust device selection criteria
- Still declare pipeline validation successful

---

## 10. Approval Signatures

| Role | Name | Decision | Date |
|------|------|----------|------|
| Data Scientist | [Pending] | GO | 2026-02-12 |
| ML Engineer | [Pending] | GO | 2026-02-12 |
| Project Lead | [Pending] | GO | 2026-02-12 |

---

## Appendix A: Quick Reference

**Device:** e0206 Cafeteria AHU  
**Signal Type:** Low-variance, continuous, weak seasonality  
**Data Quality:** Excellent (94% gold)  
**Baseline MAE:** 0.268 kW (7.5%)  
**Target MAE:** <0.240 kW (10% improvement)  
**Architecture:** Lean N-BEATS (3 generic blocks)  
**Purpose:** Pipeline validation + baseline  
**Timeline:** 2-3 days  
**Decision:** 🟢 GO

---

## Document Status

**Version:** 1.0  
**Status:** Approved  
**Next Review:** After first training run  
**Supersedes:** None (initial decision)
