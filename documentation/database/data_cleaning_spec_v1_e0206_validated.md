# Data Resampling & Cleaning Specification v1 (Validated)

## Document Status
**Version:** 1.0 e0206  
**Status:** Validated against raw device data  
**Validation Date:** 2026-02-12  
**Data Sample:** 2026-01-28 to 2026-02-11 (1,440 intervals)

---

## Scope
This specification defines the transformation of raw InfluxDB signals into a clean, 15-minute tabular dataset for modeling (N-BEATS).

**Applies to:**
* **AHU:** `e0206` (Cafeteria)
* **Bucket:** `default`
* **Measurement pattern:** `wach_e0206_*`

This document is binding for the extraction pipeline.

---

## 1. Extraction & Pivoting Strategy

### 1.1 Query Pattern

```flux
from(bucket: "default")
  |> range(start: START_TIME, stop: END_TIME)
  |> filter(fn: (r) => r["_measurement"] =~ /^wach_e0206_/)
  |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)
  |> pivot(rowKey:["_time"], columnKey: ["_measurement"], valueColumn: "_value")
```

### 1.2 Resampling Logic

**Target frequency:** 15 minutes  
**Aggregation function:** `mean`  
**Rationale:** Raw data already at 15-minute intervals; aggregation handles any sub-interval noise

**Validation Note:** Raw data shows perfect 15-minute sampling with zero gaps. Resampling serves as validation layer.

---

## 2. Column Renaming (InfluxDB → Model Variables)

The following mappings are **required** during extraction:

| InfluxDB Measurement Suffix | Model Variable Name | Notes |
|------------------------------|---------------------|-------|
| `power_total` | `kw` | Target variable |
| `power_l1` | `p_l1` | Feature |
| `power_l2` | `p_l2` | Feature |
| `power_l3` | `p_l3` | Feature |
| `current_l1` | `i_l1` | Feature |
| `current_l2` | `i_l2` | Feature |
| `current_l3` | `i_l3` | Feature |
| `volts_l1_n` | `v_l1` | Feature (L-N voltage) |
| `volts_l2_n` | `v_l2` | Feature (L-N voltage) |
| `volts_l3_n` | `v_l3` | Feature (L-N voltage) |
| `power_factor_avg` | `pf` | Feature |
| `apparent_power_total` | `kvar_tot` | Feature |
| `freq` | *(excluded)* | Not used per spec |

**Critical:** Use `volts_l*_n` (line-to-neutral), not `volts_l*_l*` (line-to-line).

---

## 3. Column Exclusion

### 3.1 Artifact Columns (Always Drop)

The following columns have no variability and must be excluded:

```python
artifact_columns = [
    'result',
    'controller', 
    'site',
    'digital_input_1_and_2',
    'table',
]
```

### 3.2 Out-of-Scope Metrics (Exclude Per Spec)

Do not extract these metrics even if present:

- Line-to-line voltages: `volts_l1_l2`, `volts_l2_l3`, `volts_l3_l1`
- Voltage averages: `volts_l_l_avg`, `volts_l_n_avg`
- Harmonic distortion: `*_thd`
- Imbalance metrics: `current_unbalance`, `volts_unbalance`
- Energy metrics: `energy_*`, `*_energy`
- Demand metrics: `*_demand`
- Reactive power: `reactive_power_*`
- Current average: `current_avg`
- Frequency: `freq`

**Rationale:** These metrics are outside Phase 1 scope (electrical modeling only).

---

## 4. Data Quality Validation

### 4.1 Range Validation

Apply bounds checking **after** extraction:

| Variable | Valid Range | Action on Violation |
|----------|-------------|---------------------|
| `kw` | [0, 10.0] | Set to NaN |
| `p_l1`, `p_l2`, `p_l3` | [0, 5.0] | Set to NaN |
| `i_l1`, `i_l2`, `i_l3` | [0, 20.0] | Set to NaN |
| `v_l1`, `v_l2`, `v_l3` | [200, 260] | Set to NaN |
| `pf` | [0, 1.0] | Set to NaN |
| `kvar_tot` | [0, 15.0] | Set to NaN |

**Note:** Negative values are physically invalid and indicate sensor error.

### 4.2 Physics Validation

Check for impossible electrical relationships:

```python
# Power factor cannot exist without both power and apparent power
if (kw > 0) and (kvar_tot > 0):
    computed_pf = kw / kvar_tot
    if abs(pf - computed_pf) > 0.2:  # 20% tolerance
        # Flag as suspicious but do not drop
        pass
```

**Action:** Log warning; do not automatically drop (allow model to learn).

---

## 5. Flatline Detection & Handling

### 5.1 Detection Rule

**Definition:** A flatline occurs when a metric has an identical value for 8+ consecutive intervals (≥2 hours).

**Metrics to check:**
- All current: `i_l1`, `i_l2`, `i_l3`
- All voltage: `v_l1`, `v_l2`, `v_l3`
- All power: `p_l1`, `p_l2`, `p_l3`, `kw`

**Detection algorithm:**

```python
def detect_flatline(series, window=8):
    """
    Returns boolean mask where True = flatline period
    """
    is_constant = series.diff().abs() < 1e-6
    flatline_mask = is_constant.rolling(window).sum() >= window
    return flatline_mask
```

### 5.2 Handling Rule

**Action:** Set flatlined values to NaN

```python
for metric in ['i_l1', 'i_l2', 'i_l3', 'v_l1', 'v_l2', 'v_l3', 'p_l1', 'p_l2', 'p_l3', 'kw']:
    flatline_mask = detect_flatline(df[metric], window=8)
    df.loc[flatline_mask, metric] = np.nan
```

**Validation Note:** Raw data shows 21-hour flatline event (2026-02-09 10:30 to 2026-02-10 07:00). This rule will correctly identify and handle it.

---

## 6. Missing Data Handling

### 6.1 Gap Detection

**Current reality:** Raw data has zero missing values. However, this may not hold for future data.

**Gap definition:** Any NaN in the dataset (post-cleaning)

### 6.2 Gap Interpolation

**Rule:** Interpolate only single missing intervals (1 gap = 15 minutes)

```python
# For each metric
for col in feature_columns:
    # Identify isolated single NaNs
    is_nan = df[col].isna()
    prev_valid = ~df[col].shift(1).isna()
    next_valid = ~df[col].shift(-1).isna()
    
    single_gap = is_nan & prev_valid & next_valid
    
    # Linear interpolation for single gaps only
    df.loc[single_gap, col] = df[col].interpolate(method='linear', limit=1)
```

**Rationale:** Single 15-minute gaps likely due to brief communication loss. Longer gaps indicate real issues.

### 6.3 Extended Gap Handling

**Rule:** Do NOT interpolate gaps ≥2 consecutive intervals (≥30 minutes)

```python
# Leave as NaN
pass
```

---

## 7. Row-Level Quality Filtering

### 7.1 Target Variable Rule

**HARD RULE:** If `kw` is NaN, **drop the row entirely**.

```python
df = df[df['kw'].notna()]
```

**Rationale:** Cannot train without target. No imputation allowed.

### 7.2 Feature Completeness Rule

**Rule:** Drop rows where >50% of features are NaN

```python
feature_columns = ['p_l1', 'p_l2', 'p_l3', 'i_l1', 'i_l2', 'i_l3', 
                   'v_l1', 'v_l2', 'v_l3', 'pf', 'kvar_tot']

nan_count = df[feature_columns].isna().sum(axis=1)
nan_ratio = nan_count / len(feature_columns)

df = df[nan_ratio <= 0.5]
```

**Validation Note:** Raw data has 0% NaN, so this rule is defensive for future data.

---

## 8. Final Dataset Characteristics

### 8.1 Schema

| Column | Data Type | Nullable | Notes |
|--------|-----------|----------|-------|
| `timestamp` | datetime64[ns, UTC] | No | Index |
| `kw` | float64 | No | Target |
| `p_l1` | float64 | Yes | Feature |
| `p_l2` | float64 | Yes | Feature |
| `p_l3` | float64 | Yes | Feature |
| `i_l1` | float64 | Yes | Feature |
| `i_l2` | float64 | Yes | Feature |
| `i_l3` | float64 | Yes | Feature |
| `v_l1` | float64 | Yes | Feature |
| `v_l2` | float64 | Yes | Feature |
| `v_l3` | float64 | Yes | Feature |
| `pf` | float64 | Yes | Feature |
| `kvar_tot` | float64 | Yes | Feature |

### 8.2 Time Features (Derived)

Add deterministic temporal features:

```python
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
```

**Note:** These are derived from timestamp, not InfluxDB.

---

## 9. Gold Dataset Definition

### 9.1 Gold Criteria

A row qualifies for the **Gold Dataset** if:

1. ✅ `kw` is present (not NaN)
2. ✅ No flatline detected in any metric
3. ✅ All metrics within valid ranges
4. ✅ ≤50% features missing
5. ✅ No physics violations (optional, warning only)

### 9.2 Quality Metrics

Track and log:

```python
gold_metrics = {
    'total_rows': len(df),
    'gold_rows': len(gold_df),
    'gold_percentage': len(gold_df) / len(df) * 100,
    'flatline_rows_dropped': flatline_drop_count,
    'target_missing_rows_dropped': target_nan_count,
    'feature_sparse_rows_dropped': sparse_row_count,
}
```

---

## 10. Pipeline Implementation Order

**Execute in this sequence:**

```
1. Extract from InfluxDB (15m resampling)
   ↓
2. Rename columns (InfluxDB → model names)
   ↓
3. Drop artifact columns
   ↓
4. Apply range validation → set violations to NaN
   ↓
5. Detect & handle flatlines → set to NaN
   ↓
6. Interpolate single-interval gaps
   ↓
7. Drop rows with target NaN
   ↓
8. Drop rows with >50% feature NaN
   ↓
9. Add time-based features
   ↓
10. Validate final schema
   ↓
11. Output Gold Dataset
```

---

## 11. Deviations from Original Draft

### 11.1 Changes After Raw Data Validation

| Original Spec | Validated Spec v1 | Reason |
|---------------|-------------------|--------|
| Generic voltage column names | Explicit `volts_l*_n` mapping | Column naming mismatch discovered |
| No flatline detection | Added 8-interval flatline rule | 21-hour flatline event found |
| Implied zero-power handling | Removed (no zeros found) | Device runs continuously (min 2.77kW) |
| Generic gap handling | Single-gap interpolation only | No gaps found; conservative approach |
| 50% NaN threshold | Kept but noted as unused | Raw data has 0% NaN |

### 11.2 What Stayed the Same

- ✅ 15-minute resampling
- ✅ Linear interpolation for small gaps
- ✅ Drop row if target missing
- ✅ 50% feature NaN threshold
- ✅ Range validation bounds
- ✅ No frequency in features

---

## 12. Known Limitations & Future Considerations

### 12.1 Device-Specific Quirks (Documented, Not "Fixed")

- **Phase imbalance:** L2 carries 44% less load than L3 (this is normal)
- **Power discrepancy:** `kw ≠ p_l1 + p_l2 + p_l3` by up to 0.5kW (acceptable)
- **No diurnal pattern:** Load varies 2.77-4.94kW independent of time of day
- **Stable frequency:** 50.00±0.1Hz (excluded from features per spec)

### 12.2 Assumptions

This spec assumes:
- Data arrives at 15-minute intervals
- Upstream aggregation already applied
- Single device, single AHU
- No equipment changes during collection period

### 12.3 Future Enhancements (Not in v1)

- Cross-device normalization
- Occupancy proxy features
- Weather integration
- Adaptive flatline thresholds per metric
- Multi-AHU comparative validation

---

## 13. Validation Checklist

Before deploying this spec:

- [ ] Column rename mapping tested on real data
- [ ] Flatline detection catches 2026-02-09 event
- [ ] Artifact columns successfully dropped
- [ ] Range validation triggers on synthetic bad data
- [ ] Target missing rows correctly dropped
- [ ] Time features correctly derived
- [ ] Final schema matches section 8.1
- [ ] Gold dataset metrics logged

---

## 14. Success Metrics

**Expected output for e0206 (2026-01-28 to 2026-02-11):**

| Metric | Expected Value |
|--------|----------------|
| Raw rows | 1,440 |
| Rows after flatline removal | ~1,357 (83 dropped) |
| Gold dataset rows | ~1,357 |
| Gold percentage | ~94% |
| Features with NaN | 0 (except during flatline) |

**Validation:** If gold percentage <90%, investigate additional data quality issues.

---

## Status

**Document Version:** 1.0 (e0206 validated)  
**Previous Version:** Draft v1 (pre-validation)  
**Validated Against:** `raw_device_data.csv` (2026-01-28 to 2026-02-11)  
**Applies To:** `extract_ahu.py` implementation  
**Next Action:** Implement pipeline and run validation test
