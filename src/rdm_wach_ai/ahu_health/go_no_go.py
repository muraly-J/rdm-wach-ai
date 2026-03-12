import pandas as pd
from pathlib import Path

############################################
# PATHS
############################################

PROJECT_ROOT = Path.cwd()

DIVERSITY_PATH = PROJECT_ROOT / "data/meta/diversity_summary.csv"
AHU_SUMMARY_PATH = PROJECT_ROOT / "data/meta/ahu_health_by_ahu.csv"
STATIONARITY_PATH = PROJECT_ROOT / "data/meta/ahu_stationarity_analysis.csv"

OUTPUT_PATH = PROJECT_ROOT / "data/meta/go_no_go_assessment.csv"

############################################
# THRESHOLDS
############################################

MIN_ANOMALY_PERCENT = 5
MIN_AHU_AFFECTED_PERCENT = 20
MIN_STATIONARY_PERCENT = 50

############################################
# COMPONENT CONFIG
############################################

COMPONENT_CONFIG = {
    "energy_score": ("energy_%", "delta_kwh"),
    "pf_score": ("pf_%", "power_factor_avg"),
    "unbalance_score": ("unbalance_%", "current_unbalance"),
    "thd_score": ("thd_%", "volts_l1_thd"),
    "overload_score": ("overload_%", "power_total")
}

############################################
# LOAD DATA
############################################

diversity_df = pd.read_csv(DIVERSITY_PATH)
ahu_summary_df = pd.read_csv(AHU_SUMMARY_PATH)
stationarity_df = pd.read_csv(STATIONARITY_PATH)

total_ahus = len(ahu_summary_df)

############################################
# ANALYSIS
############################################

results = []

for component, (ahu_col, metric) in COMPONENT_CONFIG.items():

    station_col = f"{metric}_ml_candidate"

    # anomaly %
    anomaly_row = diversity_df[diversity_df["metric"] == component]

    anomaly_percent = (
        anomaly_row["percentage"].values[0]
        if not anomaly_row.empty
        else 0
    )

    # ahu affected %
    if ahu_col in ahu_summary_df.columns:
        affected = (ahu_summary_df[ahu_col] > 1).sum()
        ahu_percent = 100 * affected / total_ahus
    else:
        ahu_percent = 0

    # stationary %
    if station_col in stationarity_df.columns:
        stationary_percent = (
            100 * stationarity_df[station_col].sum() /
            len(stationarity_df)
        )
    else:
        stationary_percent = 0

    # decision
    decision = (
        "GO"
        if (
            anomaly_percent >= MIN_ANOMALY_PERCENT
            and ahu_percent >= MIN_AHU_AFFECTED_PERCENT
            and stationary_percent >= MIN_STATIONARY_PERCENT
        )
        else "NO-GO"
    )

    results.append({
        "component": component,
        "anomaly_percent": round(anomaly_percent, 2),
        "ahu_affected_percent": round(ahu_percent, 2),
        "stationary_percent": round(stationary_percent, 2),
        "decision": decision
    })

############################################
# SAVE CSV
############################################

df = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("\nGO / NO-GO ASSESSMENT\n")
print(df)

print("\nSaved to:", OUTPUT_PATH)