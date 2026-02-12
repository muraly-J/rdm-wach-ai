# Modeling Variable List — Electrical Only (v1)

## Scope
This specification applies strictly to:
* **AHU:** `e0206` (Cafeteria)
* **Device Classification:** `C T1`
* **Data Source:** InfluxDB bucket `default`
* **Measurement Pattern:** `wach_<device_id>_<metric_name>`

This document defines the **only** variables allowed in the modeling pipeline. 
* No implicit inclusion.
* No automatic feature selection.
* No guessing.

---

## 1. Target Variable (Model Objective)
The model’s sole objective is to predict total active power (kW).

| Model Variable | Influx Metric Suffix | Data Type | Construction Logic | Missing Target Handling |
| :--- | :--- | :--- | :--- | :--- |
| **kw** | `power_total` | Float | **Primary:** Use `_value` from `wach_e0206_power_total`. <br><br> **Fallback:** If missing, compute: `power_l1 + power_l2 + power_l3` | **DROP ROW**. No imputation allowed for target. |

### Rules
* Target must never be forward-filled.
* Target must never be interpolated.
* If unavailable → row is removed.

---

## 2. Allowed Input Features (Electrical Only)
Only the following raw electrical metrics may be used as predictors. If a metric is not listed here, it is **excluded**.

| Feature Name | Influx Metric Suffix | Physical Meaning | Inclusion Logic |
| :--- | :--- | :--- | :--- |
| **p_l1** | `power_l1` | Active Power Phase 1 | Include if available |
| **p_l2** | `power_l2` | Active Power Phase 2 | Include if available |
| **p_l3** | `power_l3` | Active Power Phase 3 | Include if available |
| **i_l1** | `current_l1` | Current Phase 1 | Include |
| **i_l2** | `current_l2` | Current Phase 2 | Include |
| **i_l3** | `current_l3` | Current Phase 3 | Include |
| **v_l1** | `voltage_l1` | Voltage Phase 1 | Include |
| **v_l2** | `voltage_l2` | Voltage Phase 2 | Include |
| **v_l3** | `voltage_l3` | Voltage Phase 3 | Include |
| **pf** | `power_factor` | Power Factor | Include |
| **kvar_tot** | `apparent_power_total` | Apparent Power | Include |
| **freq** | `frequency` | Grid Frequency | **Exclude** (invariant for AHU modeling) |

---

## 3. Explicit Exclusions
The following are strictly prohibited in this phase:
* Environmental sensors (Temperature, Humidity, CO₂)
* Occupancy proxies
* External metadata
* Any metric not listed above



---

## 4. Permitted Derived Features
The following are allowed:
* Time-based deterministic features derived from timestamp:
    * Hour of day
    * Day of week
    * Weekend indicator

**Constraints:**
1.  These must be derived from the UTC timestamp index.
2.  They must not originate from InfluxDB fields.

---

## 5. Hard Rule
If a new metric appears during pivoting:
* It is excluded.
* It must be explicitly approved in a new version of this document before use.

This specification eliminates the question: *“Should I include this column?”*
**No guessing is allowed in the ML pipeline.**

---

**Status:** Active — v1  
**Applies To:** `extract_ahu.py` implementation