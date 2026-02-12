# Data Quirks & Observations — e0206 (Cafeteria)

## Document Purpose
This document records undocumented behaviors, anomalies, and device-specific characteristics discovered during raw data validation. These are **descriptive observations**, not prescriptive rules.

**Date:** 2026-02-12  
**Data Period:** 2026-01-28 to 2026-02-11 (15 days)  
**Total Samples:** 1,440 (15-minute intervals)

---

## 1. Sensor Flatline Event

### Critical Finding
**All electrical metrics** experienced synchronized flatline from **2026-02-09 10:30 to 2026-02-10 07:00** (20.75 hours / 83 intervals).

**Affected metrics:**
- `current_l1`: flatlined at 7.4667A
- `current_l2`: flatlined at 5.6167A  
- `current_l3`: flatlined at 10.5433A
- `volts_l1_n`: flatlined at 234.8833V
- `volts_l2_n`: flatlined at 233.5000V
- `volts_l3_n`: flatlined at 235.0833V
- `power_l2`: flatlined at 0.9198kW

**Interpretation:**
- Sensor/communication failure, not physical reality
- Values locked at specific readings
- Duration: 21 hours continuous
- All metrics show exact same value for entire period

**Impact:** These 83 intervals must be flagged and excluded from training data.

**Additional Observation:**
- `current_l2` shows a second, shorter flatline: 8 intervals (2 hours) on 2026-02-10 from 15:00-16:45 at 5.6200A

---

## 2. Severe Phase Imbalance

### Current Distribution
**Not evenly distributed across phases:**

| Phase | Mean Current | Relative to Max |
|-------|--------------|-----------------|
| L1    | 6.31A        | 66.5%           |
| L2    | 5.62A        | **59.2% ← Lowest** |
| L3    | 9.49A        | 100%            |

**Current imbalance statistics:**
- Mean: 44.50%
- Max: 54.32%
- Min: 35.32%

### Power Distribution
**Phase power follows current pattern:**

| Phase | Mean Power | Percentage |
|-------|------------|------------|
| L1    | 1.13kW     | 31.4%      |
| L2    | 0.92kW     | **25.6% ← Lowest** |
| L3    | 1.55kW     | 43.0%      |

**Interpretation:**
- L2 consistently carries less load
- Could indicate:
  - Intentional load distribution
  - Equipment wiring configuration
  - Single-phase loads on L1 and L3
- This is **stable behavior**, not a sensor issue

**Implication:** Phase imbalance is normal for this device. Models should learn this pattern.

---

## 3. Power Calculation Discrepancy

### power_total vs Sum of Phases

**Finding:** `power_total` does NOT exactly equal `power_l1 + power_l2 + power_l3`

| Statistic | Value (kW) |
|-----------|------------|
| Mean difference | -0.0011 |
| Std difference | 0.0230 |
| Max difference | +0.1240 |
| Min difference | -0.5052 |

**Typical error:** ±0.02kW (< 1%)  
**Worst case:** 0.5kW (14% at minimum load)

**Interpretation:**
- `power_total` likely measured independently at main connection
- Phase powers summed from individual sensors
- Small calibration differences between sensors
- Larger relative errors at low power

**Implication:** 
- Use `power_total` as ground truth (target variable)
- Do NOT compute `power_total` from phases
- Discrepancy is acceptable and expected

---

## 4. Column Naming Inconsistency

### Spec vs Reality Mapping

The InfluxDB measurement names differ from variable list specification:

| Spec Name | Actual Column Name | Status |
|-----------|-------------------|--------|
| `voltage_l1` | `volts_l1_n` | **Rename required** |
| `voltage_l2` | `volts_l2_n` | **Rename required** |
| `voltage_l3` | `volts_l3_n` | **Rename required** |
| `power_factor` | `power_factor_avg` | **Rename required** |
| `frequency` | `freq` | **Rename required** |

**Action:** Extraction pipeline must map these during pivoting.

---

## 5. Artifact Columns

### Irrelevant Columns Present in CSV
The following columns have **no variability** and should be ignored:

| Column | Unique Values | Notes |
|--------|---------------|-------|
| `result` | 1 (always 0.0) | Artifact |
| `controller` | 1 (always 0.0) | Artifact |
| `site` | 1 (always 0.0) | Artifact |
| `digital_input_1_and_2` | 1 (always 0.0) | Artifact |
| `table` | 696 | Unexplained floating values |

**Action:** Filter these out during extraction.

---

## 6. Extra Metrics Not in Spec

### Metrics Present but Excluded from Model

The raw data contains **additional metrics** not included in the variable list:

**Voltage variants:**
- `volts_l1_l2`, `volts_l2_l3`, `volts_l3_l1` (line-to-line)
- `volts_l_l_avg`, `volts_l_n_avg` (averages)
- `volts_l1_thd`, `volts_l2_thd`, `volts_l3_thd` (total harmonic distortion)
- `volts_unbalance`

**Current variants:**
- `current_avg`
- `current_l1_thd`, `current_l3_thd` (harmonic distortion)
- `current_unbalance`

**Energy/demand metrics:**
- `energy_import`, `energy_export`
- `apparent_energy`
- `reactive_energy_import`, `reactive_energy_export`
- `power_demand`, `max_power_demand`, `apparent_power_demand`
- `reactive_power_demand`

**Reactive power:**
- `reactive_power_l1`, `reactive_power_l2`, `reactive_power_l3`
- `reactive_power_total`

**Action:** These are excluded per spec. Document for future phases.

---

## 7. No True Zero Power Periods

### Shutdown Behavior

**Finding:** This device does **not shut down**.

| Metric | Value |
|--------|-------|
| Minimum power_total | 2.77kW |
| Hours below 2.8kW | 1.25 hours (0.3% of time) |
| True zero periods | 0 |

**Interpretation:**
- Cafeteria AHU runs continuously
- No overnight shutdown
- Minimum load ~2.8kW (likely base load: controls, fans at minimum speed)
- Low power periods occur randomly, not on schedule

**Implication:** 
- Zero-power handling rules in spec are unnecessary for this device
- No need for shutdown detection logic
- Power range: 2.77kW to 4.94kW

---

## 8. Stable Frequency

### Grid Frequency Behavior

| Metric | Value |
|--------|-------|
| Mean | 50.00Hz |
| Std | 0.0425Hz |
| Min | 49.90Hz |
| Max | 50.10Hz |

**Interpretation:**
- Grid frequency is extremely stable
- Variation within ±0.1Hz
- As expected for Malaysian grid

**Implication:** 
- Confirms spec decision to exclude `frequency` from modeling
- No informational value for load prediction

---

## 9. No Missing Data

### Data Completeness

**Finding:** Raw dataset has **zero missing values** in electrical metrics.

- All 1,440 rows complete
- No NaN in any primary metric
- Perfect 15-minute sampling (no gaps or duplicates)

**Interpretation:**
- This is **exceptionally clean** data
- Aggregation/averaging likely already performed upstream
- The flatline event (Section 1) is not "missing" data but frozen sensor values

**Implication:**
- Most data cleaning rules in spec are **not needed** for this device
- Gap interpolation rules may be unnecessary
- NaN threshold (50%) will never trigger

---

## 10. Power Factor Behavior

### Normal Range

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| `power_factor_avg` | 0.64 | 0.81 | 0.71 |
| `power_factor_l1` | 0.51 | 0.90 | 0.73 |
| `power_factor_l2` | 0.68 | 0.79 | 0.70 |
| `power_factor_l3` | 0.64 | 0.77 | 0.69 |

**Observations:**
- All values within valid range [0, 1]
- No invalid values (>1 or <-1)
- L1 shows widest variation (0.51-0.90)
- L2 and L3 more stable

**Interpretation:**
- Typical inductive loads (motors, fans)
- L1 variability suggests variable speed drive or switching loads
- No power factor correction system active

---

## 11. Voltage Stability

### Voltage Ranges (L-N)

| Phase | Min (V) | Max (V) | Mean (V) | Std (V) |
|-------|---------|---------|----------|---------|
| L1    | 231.10  | 240.10  | 235.93   | 1.76    |
| L2    | 229.60  | 238.62  | 234.37   | 1.76    |
| L3    | 231.28  | 240.30  | 236.17   | 1.81    |

**Observations:**
- All phases within ±4V of nominal 230V
- Very stable (std ~1.8V)
- L2 consistently ~1.5V lower than L1/L3
- No voltage sags or spikes detected

**Interpretation:**
- Grid supply is stable
- Slight phase voltage differences normal
- Well within acceptable ±10% tolerance

---

## 12. No Diurnal Pattern

### Load Profile

**Finding:** Power consumption shows **minimal time-of-day variation**.

| Hour | Mean (kW) | Std (kW) |
|------|-----------|----------|
| Night (0-6) | 3.54 | 0.55 |
| Day (7-18) | 3.60 | 0.56 |
| Evening (19-23) | 3.64 | 0.54 |

**Observation:**
- Load range: 2.77-4.94kW across all hours
- No clear occupancy signal
- Slight evening peak (~3.7kW) but not pronounced

**Interpretation:**
- Cafeteria AHU may serve 24-hour facility
- Or: operates independently of kitchen hours
- Suggests minimal occupancy-driven load variation

**Implication:** Time-based features may have limited predictive value for this device.

---

## Summary Table: Key Quirks

| # | Quirk | Impact | Action |
|---|-------|--------|--------|
| 1 | 21-hour flatline event | Training data corruption | Flag 83 intervals for exclusion |
| 2 | 44% phase imbalance | Normal for device | Allow in model |
| 3 | power_total ≠ sum(phases) | ±0.5kW max difference | Use power_total as-is |
| 4 | Column name mismatches | Extraction complexity | Rename during pivot |
| 5 | Zero missing data | Cleaning rules unused | Simplify validation |
| 6 | No shutdowns | Power always >2.7kW | Remove zero-handling logic |
| 7 | Stable frequency | No variation | Confirm exclusion |
| 8 | L2 always lowest load | 25% vs 31%/43% | Document, allow |
| 9 | 20+ extra columns | Not in spec | Explicitly drop |
| 10 | No diurnal pattern | Continuous operation | Time features may be weak |

---

## Status

**Document Version:** 1.0  
**Validated Against:** `/mnt/user-data/uploads/raw_device_data.csv`  
**Next Action:** Update data cleaning spec based on these findings
