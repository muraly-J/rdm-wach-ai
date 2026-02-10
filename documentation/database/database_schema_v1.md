# Database Schema v1 — InfluxDB (WACH)

## Bucket
**wach_bucket_3**

---

## Measurements

- Metric-specific measurements per device  
- There is **no single unified AHU measurement**
- Each physical or logical signal is stored as its own measurement

**Examples (power-related):**
- `wach_e0101_power_l1`
- `wach_e0101_power_l2`
- `wach_e0101_power_l3`
- `wach_e0101_power_total`

---

## Tags

The following tags are present on measurements:

- `controller` (string)  
- `site` (string)  
- `units` (string)

> ⚠️ No semantic tags (e.g. OT / Ward, building block) are currently stored in InfluxDB.

---

## Fields

- `value` (float)  
  - Numeric value for the given metric

---

## Time

- All timestamps are stored in **UTC**

---

## Implicit Dimensions (Derived During Extraction)

The following dimensions are **not explicitly stored** in InfluxDB and must be **derived from measurement names** during data extraction.

### Measurement Naming Pattern

All measurements follow a structured naming convention: wach_<device_id>_<metric_name>

**Examples:**
- `wach_e0101_power_l1`
- `wach_e0101_power_l2`
- `wach_e0101_power_l3`
- `wach_e0101_power_total`

---

### Derived Dimensions

- `device_id` (used as `ahu_id`)  
  - Parsed from the measurement name  
  - Represents the physical AHU or electrical device  
  - Example:
    - `wach_e0101_power_l1` → `ahu_id = e0101`

- `metric_name`  
  - Parsed from the measurement name suffix  
  - Represents the physical quantity being measured  
  - Example:
    - `wach_e0101_power_l1` → `metric_name = power_l1`

---

## Notes & Implications

- All measurements are **metric-centric**, not AHU-centric
- AHU-level datasets must be constructed in the extraction layer by:
  - grouping on derived `ahu_id`
  - pivoting selected `metric_name` values into columns
- No semantic classification (e.g. OT vs Ward, building block) exists in InfluxDB
- Environmental and location metadata must be introduced via:
  - external mapping tables
  - PostgreSQL integration (Phase 2+)
- Downstream ML pipelines must convert metric-centric data into model-ready tabular datasets

---

## Status

- Schema reflects **current InfluxDB reality**
- This document is the authoritative reference for:
  - data extraction
  - dataset construction
  - Phase 1 ML development