## 📌 PR Title

`[area] concise description`

**Example:**  
`[data] add 15-min resampling pipeline`

---

## 🔍 What does this PR do?

Brief, factual description of what changed and why.

- No speculation
- No future plans
- Just what this PR actually delivers

---

## 📂 Scope

- Data extraction
- Data cleaning / resampling
- Dataset export
- Model training
- InfluxDB write-back
- Residual / alert logic
- Documentation

---

## 📜 Specification followed

Link or reference the spec or instruction this PR implements:

- Data spec v1
- Modeling instruction
- Verbal instruction (include date)
- Other (describe):

---

## ⚠️ Assumptions made

List all explicit assumptions.  
If none, write **“None”**.

**Examples:**
- Forward-fill limited to 1 interval
- UTC timestamps enforced
- Missing AHUs excluded from training

---

## 🧪 How was this tested?

- Local run
- Sample AHU
- Visual inspection
  Metrics check

**Details:**
- Dataset size tested:
- Time range:
- Key validation performed:

---

## 📦 Artifacts produced

List files or outputs created or modified by this PR.

**Examples:**
- `data/gold/ahu_15min.parquet`
- `scripts/resample.py`
- Training metrics CSV
- Diagnostic plots

---

## 🚫 What this PR does NOT do

(Explicitly list exclusions to prevent scope creep)

**Examples:**
- Does not tune model hyperparameters
- Does not change InfluxDB schema
- Does not affect alert thresholds
