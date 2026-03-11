import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning

warnings.simplefilter("ignore", InterpolationWarning)

############################################
# CONFIG
############################################

PROJECT_ROOT = Path.cwd()

DATA_PATH = PROJECT_ROOT / "data" / "merged" / "ahu_all.parquet"

OUTPUT_DETAIL = PROJECT_ROOT / "data" / "meta" / "ahu_stationarity_analysis.csv"
OUTPUT_SUMMARY = PROJECT_ROOT / "data" / "meta" / "metric_ml_summary.csv"

KEY_METRICS = [
    "delta_kwh",
    "power_factor_avg",
    "current_unbalance",
    "composite_thd_24h",
    "power_total"
]

MIN_VARIANCE = 0.001
MAX_POINTS = 5000

############################################
# LOAD DATA
############################################

df = pd.read_parquet(DATA_PATH)

if "time" in df.columns:
    df = df.rename(columns={"time": "timestamp"})

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

df = df.sort_values(["ahu_id", "timestamp"])

############################################
# CREATE ENERGY DELTA
############################################

if "delta_kwh" not in df.columns and "energy_import" in df.columns:
    df["delta_kwh"] = df.groupby("ahu_id")["energy_import"].diff().fillna(0)

############################################
# STATIONARITY TEST
############################################

def test_stationarity(series):

    series = series.dropna()

    if len(series) < 20:
        return False, np.nan, np.nan

    if len(series) > MAX_POINTS:
        step = int(len(series) / MAX_POINTS)
        series = series.iloc[::step]

    try:
        adf_p = adfuller(series)[1]
    except:
        adf_p = np.nan

    try:
        kpss_p = kpss(series, regression="c")[1]
    except:
        kpss_p = np.nan

    stationary = (adf_p < 0.05) and (kpss_p > 0.05)

    return stationary, adf_p, kpss_p

############################################
# ANALYSIS
############################################

results = []

for ahu_id, g in df.groupby("ahu_id"):

    row = {"ahu_id": ahu_id}

    for metric in KEY_METRICS:

        if metric not in g.columns:
            row[f"{metric}_stationary"] = False
            row[f"{metric}_variance"] = np.nan
            row[f"{metric}_ml_candidate"] = False
            continue

        series = g[metric]

        stationary, adf_p, kpss_p = test_stationarity(series)

        variance = series.var()

        ml_candidate = stationary and variance >= MIN_VARIANCE

        row[f"{metric}_stationary"] = stationary
        row[f"{metric}_variance"] = variance
        row[f"{metric}_ml_candidate"] = ml_candidate

    results.append(row)

result_df = pd.DataFrame(results)

############################################
# SAVE DETAILED RESULT
############################################

result_df.to_csv(OUTPUT_DETAIL, index=False)

############################################
# CREATE METRIC SUMMARY
############################################

summary_rows = []

for metric in KEY_METRICS:

    candidates = result_df[f"{metric}_ml_candidate"].sum()

    total = len(result_df)

    summary_rows.append({
        "metric": metric,
        "AHUs_valid_for_ML": candidates,
        "total_AHUs": total,
        "percentage": round(100*candidates/total,2)
    })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(OUTPUT_SUMMARY, index=False)

print("Analysis complete")
print("Detailed result:", OUTPUT_DETAIL)
print("Summary result:", OUTPUT_SUMMARY)