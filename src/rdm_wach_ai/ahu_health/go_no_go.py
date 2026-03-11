import pandas as pd
import numpy as np
from pathlib import Path

############################################
# CONFIG
############################################

PROJECT_ROOT = Path.cwd()

DIVERSITY_PATH = PROJECT_ROOT / "data" / "meta" / "diversity_summary.csv"
AHU_SUMMARY_PATH = PROJECT_ROOT / "data" / "meta" / "ahu_health_by_ahu.csv"
STATIONARITY_PATH = PROJECT_ROOT / "data" / "meta" / "ahu_stationarity_analysis.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "meta" / "go_no_go_assessment.csv"

# Thresholds
MIN_ANOMALY_PERCENT = 5       # % of flagged hours for a component to be ML-feasible
MIN_AHU_ML_PERCENT = 50       # % of AHUs valid for ML per metric
MIN_VARIANCE = 0.001          # minimal variance to consider informative

# Components (score columns from your previous outputs)
COMPONENTS = ["energy_score","pf_score","unbalance_score","thd_score","overload_score"]

############################################
# LOAD DATA
############################################

diversity_df = pd.read_csv(DIVERSITY_PATH)
ahu_summary_df = pd.read_csv(AHU_SUMMARY_PATH)
stationarity_df = pd.read_csv(STATIONARITY_PATH)

total_ahus = len(ahu_summary_df)

############################################
# FUNCTION TO CALCULATE GO/NO-GO
############################################

def compute_go_no_go(component):
    # 1️⃣ Diversity info (global % unhealthy)
    diversity_row = diversity_df[diversity_df["metric"]==component]
    anomaly_percent = float(diversity_row["percentage"].values[0])

    # 2️⃣ % of AHUs valid for ML
    ahu_col = component + "_flag" if component != "overload_score" else "overload_%"
    
    if ahu_col in ahu_summary_df.columns:
        if ahu_col.endswith("_%"):
            ahu_ml_percent = ahu_summary_df[ahu_col].mean()  # already in %
        else:
            ahu_ml_percent = 100 * ahu_summary_df[ahu_col].sum() / total_ahus
    else:
        ahu_ml_percent = 0

    # 3️⃣ Stationarity & variance check (optional, use stationarity_df)
    # Here, simple check: % of AHUs with stationary & variance >= MIN_VARIANCE
    metric_stationary = component.replace("_score","")
    if f"{metric_stationary}_ml_candidate" in stationarity_df.columns:
        stationary_percent = 100 * stationarity_df[f"{metric_stationary}_ml_candidate"].sum() / total_ahus
    else:
        stationary_percent = 0

    # 4️⃣ Make Go/No-Go decision
    go = (anomaly_percent >= MIN_ANOMALY_PERCENT) and \
         (ahu_ml_percent >= MIN_AHU_ML_PERCENT) and \
         (stationary_percent >= MIN_AHU_ML_PERCENT)

    decision = "GO" if go else "NO-GO"

    return {
        "component": component,
        "anomaly_percent": round(anomaly_percent,2),
        "ahu_ml_percent": round(ahu_ml_percent,2),
        "stationary_percent": round(stationary_percent,2),
        "decision": decision
    }

############################################
# CREATE ASSESSMENT
############################################

assessment_rows = [compute_go_no_go(c) for c in COMPONENTS]

assessment_df = pd.DataFrame(assessment_rows)

# Save
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
assessment_df.to_csv(OUTPUT_PATH, index=False)

print("Go/No-Go ML assessment completed!")
print("Saved to:", OUTPUT_PATH)
print("\nPreview:\n", assessment_df)