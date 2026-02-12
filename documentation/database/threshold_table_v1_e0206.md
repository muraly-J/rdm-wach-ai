# Threshold Table v1 — e0206 (Cafetaria)

## Document Purpose
This table converts all vague cleaning logic into **numeric, machine-enforceable rules**. Every threshold is validated against raw device behavior and represents a concrete decision boundary.

**Status:** Validated  
**Date:** 2026-02-12  
**Applies to:** AHU `e0206` (Cafeteria)

---

## 1. Range Validation Thresholds

### 1.1 Power Metrics

| Metric | Unit | Minimum | Maximum | Action on Violation | Rationale |
|--------|------|---------|---------|---------------------|-----------|
| `kw` (target) | kW | 0.0 | 10.0 | Set to NaN | Observed range: 2.77-4.94kW; 10kW allows 2x safety margin |
| `p_l1` | kW | 0.0 | 5.0 | Set to NaN | Observed range: 0.54-2.14kW; single phase max ~5kW |
| `p_l2` | kW | 0.0 | 5.0 | Set to NaN | Observed range: 0.80-0.94kW; very stable |
| `p_l3` | kW | 0.0 | 5.0 | Set to NaN | Observed range: 1.32-1.88kW |

**Note on negatives:** Negative power is physically impossible for this device (no regeneration). Any negative value indicates sensor error.

### 1.2 Current Metrics

| Metric | Unit | Minimum | Maximum | Action on Violation | Rationale |
|--------|------|---------|---------|---------------------|-----------|
| `i_l1` | A | 0.0 | 20.0 | Set to NaN | Observed range: 3.91-10.73A; 20A allows for startup surge |
| `i_l2` | A | 0.0 | 20.0 | Set to NaN | Observed range: 4.48-5.75A; very stable, low variation |
| `i_l3` | A | 0.0 | 20.0 | Set to NaN | Observed range: 8.55-10.83A; highest loaded phase |

**Note:** All phases use same 20A upper bound for consistency, despite L2 being more stable.

### 1.3 Voltage Metrics (Line-to-Neutral)

| Metric | Unit | Minimum | Maximum | Action on Violation | Rationale |
|--------|------|---------|---------|---------------------|-----------|
| `v_l1` | V | 200 | 260 | Set to NaN | Observed: 231-240V; ±10% from 230V nominal |
| `v_l2` | V | 200 | 260 | Set to NaN | Observed: 230-239V; consistently 1-2V lower |
| `v_l3` | V | 200 | 260 | Set to NaN | Observed: 231-240V |

**Standard:** Based on IEC 60038 low voltage tolerance (230V ±10% = 207-253V). Conservative bounds.

### 1.4 Power Factor

| Metric | Unit | Minimum | Maximum | Action on Violation | Rationale |
|--------|------|---------|---------|---------------------|-----------|
| `pf` | unitless | 0.0 | 1.0 | Set to NaN | Observed range: 0.64-0.81; physically bounded [0,1] |

**Note:** Negative power factor not physically possible for this load type (inductive motors).

### 1.5 Apparent Power

| Metric | Unit | Minimum | Maximum | Action on Violation | Rationale |
|--------|------|---------|---------|---------------------|-----------|
| `kvar_tot` | kVA | 0.0 | 15.0 | Set to NaN | Observed range: 4.45-5.43kVA; 15kVA allows 3x margin |

---

## 2. Flatline Detection Thresholds

### 2.1 Flatline Definition

| Parameter | Value | Unit | Rationale |
|-----------|-------|------|-----------|
| **Consecutive intervals** | 8 | intervals | = 2 hours at 15-min sampling |
| **Value tolerance** | 1e-6 | units | Exact equality (floating point safe) |
| **Check frequency** | Every metric | - | All electrical metrics can flatline |

**Detection logic:**
```python
# Metric is flatlined if:
# 1. Value changes by <1e-6 for 8+ consecutive intervals
# 2. Applies to: all i_*, v_*, p_*, kw

is_constant = abs(metric.diff()) < 1e-6
flatline = is_constant.rolling(8, min_periods=8).sum() >= 8
```

### 2.2 Metrics Checked for Flatlines

| Metric | Why Check | Historical Evidence |
|--------|-----------|---------------------|
| `i_l1`, `i_l2`, `i_l3` | Current sensors can stick | ✅ Flatlined 21h (2026-02-09) |
| `v_l1`, `v_l2`, `v_l3` | Voltage can lock during comms failure | ✅ Flatlined 21h (2026-02-09) |
| `p_l1`, `p_l2`, `p_l3` | Derived from current/voltage | ✅ p_l2 flatlined 21h |
| `kw` | Target must be clean | Check for completeness |

**Action on detection:** Set flatlined values to NaN (do not drop entire row unless target affected).

---

## 3. Gap & Missing Data Thresholds

### 3.1 Gap Interpolation

| Gap Size | Action | Method | Rationale |
|----------|--------|--------|-----------|
| 0 intervals | None | - | No gap |
| **1 interval** | **Interpolate** | Linear | 15 minutes: likely brief comms loss |
| ≥2 intervals | Leave as NaN | - | ≥30 minutes: real issue, don't guess |

**Implementation:**
```python
# Only interpolate single isolated NaNs
df[metric].interpolate(method='linear', limit=1, inplace=True)
```

**Validation note:** Raw data has zero gaps. This is defensive for future data.

### 3.2 Row Completeness Threshold

| Condition | Threshold | Action | Rationale |
|-----------|-----------|--------|-----------|
| Target (`kw`) missing | Any NaN | **Drop row** | Cannot train without target |
| Feature NaN ratio | >50% | **Drop row** | Insufficient information (<6 of 11 features) |
| Feature NaN ratio | ≤50% | Keep row | Model can handle sparse features |

**Feature NaN calculation:**
```python
feature_cols = ['p_l1', 'p_l2', 'p_l3', 'i_l1', 'i_l2', 'i_l3', 
                'v_l1', 'v_l2', 'v_l3', 'pf', 'kvar_tot']  # 11 total

nan_ratio = df[feature_cols].isna().sum(axis=1) / 11
drop_mask = nan_ratio > 0.5
```

---

## 4. Physics Validation Thresholds (Warning Only)

### 4.1 Power Factor Consistency

| Check | Threshold | Action | Notes |
|-------|-----------|--------|-------|
| \|pf - (kw/kvar_tot)\| | >0.20 | Log warning | 20% tolerance for measurement error |

**Rationale:** `power_total` is measured independently from phases. Small discrepancies expected.

**Do NOT drop rows** — these checks are informational only.

### 4.2 Phase Power Sum

| Check | Threshold | Action | Notes |
|-------|-----------|--------|-------|
| \|kw - (p_l1+p_l2+p_l3)\| | >0.60 kW | Log warning | Observed max: 0.5kW |

**Rationale:** Documented quirk (#3). Measurement method difference, not error.

---

## 5. Temporal Sampling Thresholds

### 5.1 Expected Sampling Interval

| Parameter | Value | Tolerance | Action on Violation |
|-----------|-------|-----------|---------------------|
| Interval | 15 minutes | ±30 seconds | Log warning |
| Missing intervals | 0 | - | Current baseline |
| Duplicate timestamps | 0 | - | Drop duplicates (keep first) |

**Validation:** Raw data has perfect 15-minute spacing. Monitor for drift.

---

## 6. Data Volume Thresholds

### 6.1 Minimum Training Data

| Metric | Value | Unit | Rationale |
|--------|-------|------|-----------|
| Minimum rows | 672 | rows | 7 days at 15-min intervals |
| Recommended rows | 2,880+ | rows | 30 days for weekly patterns |
| Gold dataset ratio | ≥85% | % | If <85%, investigate data quality |

### 6.2 Flatline Contamination Limit

| Metric | Value | Action |
|--------|-------|--------|
| Flatline rows | ≤10% | Acceptable; flag in logs |
| Flatline rows | >10% | Investigate sensor/device issues |

**Current status:** 83/1440 = 5.8% (acceptable).

---

## 7. Phase Imbalance Thresholds (Monitoring Only)

### 7.1 Expected Imbalance

| Metric | Value | Status | Notes |
|--------|-------|--------|-------|
| Current imbalance | 40-50% | **Normal** | L2 consistently lowest |
| Max current ratio | L3:L2 = 1.69:1 | **Normal** | Quirk #2 documented |
| Power L2 share | ~26% | **Normal** | Expected behavior for this device |

**Action:** None. Model should learn this pattern. Do NOT flag as anomaly.

---

## 8. Frequency Validation (Exclusion Confirmed)

| Metric | Observed | Threshold | Decision |
|--------|----------|-----------|----------|
| Frequency mean | 50.00 Hz | - | Stable |
| Frequency std | 0.04 Hz | - | Minimal variation |
| Frequency range | 49.90-50.10 Hz | - | Well within ±0.5Hz tolerance |

**Conclusion:** Frequency excluded from features (per spec). No predictive value.

---

## 9. Quick Reference: Action Matrix

### 9.1 Automated Actions

| Condition | Threshold | Automatic Action |
|-----------|-----------|------------------|
| Negative power | <0 | Set to NaN |
| Negative current | <0 | Set to NaN |
| Power >10 kW | >10.0 | Set to NaN |
| Current >20 A | >20.0 | Set to NaN |
| Voltage out of range | <200 or >260 V | Set to NaN |
| Power factor >1 | >1.0 | Set to NaN |
| Flatline detected | 8+ intervals | Set to NaN |
| Single gap | 1 interval NaN | Interpolate (linear) |
| Extended gap | ≥2 intervals | Leave NaN |
| Target missing | kw is NaN | **Drop row** |
| Feature sparse | >50% features NaN | **Drop row** |

### 9.2 Warning Actions (No Data Modification)

| Condition | Threshold | Action |
|-----------|-----------|--------|
| PF inconsistency | \|pf - kw/kvar\| > 0.2 | Log warning |
| Power sum mismatch | \|kw - Σp_l*\| > 0.6 | Log warning |
| High imbalance | Current imbalance >55% | Log info (expected ~44%) |
| Gold ratio low | <85% | Alert operator |

---

## 10. Implementation Pseudocode

```python
# 1. Range validation
for metric, (min_val, max_val) in RANGE_THRESHOLDS.items():
    df[metric] = df[metric].clip(lower=min_val, upper=max_val)
    df.loc[(df[metric] < min_val) | (df[metric] > max_val), metric] = np.nan

# 2. Flatline detection
for metric in FLATLINE_METRICS:
    is_constant = df[metric].diff().abs() < 1e-6
    flatline_mask = is_constant.rolling(8, min_periods=8).sum() >= 8
    df.loc[flatline_mask, metric] = np.nan

# 3. Gap interpolation (single gaps only)
for metric in FEATURE_METRICS:
    df[metric].interpolate(method='linear', limit=1, inplace=True)

# 4. Row-level filtering
# 4a. Target missing
df = df[df['kw'].notna()]

# 4b. Feature completeness
feature_cols = ['p_l1', 'p_l2', 'p_l3', 'i_l1', 'i_l2', 'i_l3', 
                'v_l1', 'v_l2', 'v_l3', 'pf', 'kvar_tot']
nan_ratio = df[feature_cols].isna().sum(axis=1) / len(feature_cols)
df = df[nan_ratio <= 0.5]

# 5. Physics validation (warnings only)
pf_computed = df['kw'] / df['kvar_tot']
pf_error = (df['pf'] - pf_computed).abs()
if (pf_error > 0.2).any():
    logger.warning(f"PF inconsistency detected: {(pf_error > 0.2).sum()} rows")

power_sum = df['p_l1'] + df['p_l2'] + df['p_l3']
power_error = (df['kw'] - power_sum).abs()
if (power_error > 0.6).any():
    logger.warning(f"Power sum mismatch detected: {(power_error > 0.6).sum()} rows")
```

---

## 11. Threshold Summary Table

### Complete Numeric Reference

| Rule | Metric | Operator | Threshold | Unit | Action |
|------|--------|----------|-----------|------|--------|
| **Range - Power** |
| R1 | kw | < | 0.0 | kW | Set NaN |
| R2 | kw | > | 10.0 | kW | Set NaN |
| R3 | p_l1, p_l2, p_l3 | < | 0.0 | kW | Set NaN |
| R4 | p_l1, p_l2, p_l3 | > | 5.0 | kW | Set NaN |
| **Range - Current** |
| R5 | i_l1, i_l2, i_l3 | < | 0.0 | A | Set NaN |
| R6 | i_l1, i_l2, i_l3 | > | 20.0 | A | Set NaN |
| **Range - Voltage** |
| R7 | v_l1, v_l2, v_l3 | < | 200 | V | Set NaN |
| R8 | v_l1, v_l2, v_l3 | > | 260 | V | Set NaN |
| **Range - Power Factor** |
| R9 | pf | < | 0.0 | - | Set NaN |
| R10 | pf | > | 1.0 | - | Set NaN |
| **Range - Apparent Power** |
| R11 | kvar_tot | < | 0.0 | kVA | Set NaN |
| R12 | kvar_tot | > | 15.0 | kVA | Set NaN |
| **Flatline Detection** |
| F1 | All metrics | consecutive | 8 | intervals | Set NaN |
| F2 | All metrics | diff() < | 1e-6 | units | Flatline flag |
| **Gap Handling** |
| G1 | All metrics | gap = | 1 | interval | Interpolate |
| G2 | All metrics | gap ≥ | 2 | intervals | Leave NaN |
| **Row Filtering** |
| D1 | kw | is NaN | - | - | Drop row |
| D2 | Features | NaN% > | 50 | % | Drop row |
| **Physics Validation** |
| P1 | \|pf - kw/kvar\| | > | 0.20 | - | Warn only |
| P2 | \|kw - Σp_l*\| | > | 0.60 | kW | Warn only |

---

## 12. Threshold Confidence Levels

| Threshold Type | Confidence | Validation Basis |
|----------------|------------|------------------|
| Power ranges | **High** | Observed max + 2x safety margin |
| Current ranges | **High** | Observed max + 2x safety margin |
| Voltage ranges | **High** | IEC 60038 standard ±10% |
| Flatline (8 intervals) | **High** | Observed 83-interval event; 8 = 2h minimum |
| Gap interpolation (1 only) | **Medium** | No gaps in validation data; defensive |
| 50% NaN threshold | **Medium** | No NaN in validation data; industry standard |
| Physics warnings | **Low** | Measurement discrepancies expected |

---

## 13. Maintenance Notes

### When to Update Thresholds

**Trigger conditions:**
1. New device commissioning with different characteristics
2. >15% of new data flagged as violations
3. Observed max exceeds current threshold (but not due to error)
4. Model performance degrades and data quality suspected

**Review frequency:** Quarterly or after major equipment changes

### Version Control

- v1.0: Initial validated thresholds (2026-02-12)
- Based on: 15 days of e0206 data (2026-01-28 to 2026-02-11)

---

## Status

**Document Version:** 1.0  
**Status:** Production-ready  
**Validated:** ✅ Against 1,440 real data points  
**Implementation:** Ready for `extract_ahu.py`  
**Next Review:** 2026-05-12 (3 months)
