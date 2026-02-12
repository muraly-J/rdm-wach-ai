# Data Resampling & Cleaning Specification (v1)

## Scope
This specification defines the transformation of raw InfluxDB signals into a clean, 15-minute tabular dataset for modeling (N-BEATS).

**Applies to:**
* **AHU:** `e0206` (Cafeteria)
* **Bucket:** `default`
* **Measurement pattern:** `wach_e0206_*`

This document is binding for the extraction pipeline.

---

## 1. Extraction & Pivoting Strategy

### Query Pattern
```flux
from(bucket)
  |> range(...)
  |> filter(measurement =~ /wach_e0206_/)