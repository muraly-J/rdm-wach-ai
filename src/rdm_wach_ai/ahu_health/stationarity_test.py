import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning

# Suppress warnings
warnings.simplefilter("ignore", InterpolationWarning)
warnings.simplefilter("ignore", RuntimeWarning)

############################################
# CONFIG
############################################

PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / "data/merged/ahu_clean.parquet"
OUTPUT_DETAIL = PROJECT_ROOT / "data/meta/ahu_stationarity_analysis.csv"
OUTPUT_SUMMARY = PROJECT_ROOT / "data/meta/metric_ml_summary.csv"

MIN_VARIANCE = 0.001
MAX_POINTS = 5000
RESAMPLE_FREQ = "1h"  # lowercase h

############################################
# LOAD DATA
############################################

df = pd.read_parquet(DATA_PATH)
if "time" in df.columns:
    df = df.rename(columns={"time": "timestamp"})

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values(["ahu_id", "timestamp"]).copy()

# Create delta_kwh if not exists
if "delta_kwh" not in df.columns and "energy_import" in df.columns:
    df["delta_kwh"] = df.groupby("ahu_id")["energy_import"].diff().fillna(0)

# Detect numeric metrics
exclude_cols = ["ahu_id", "timestamp"]
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
KEY_METRICS = [c for c in numeric_cols if c not in exclude_cols]

print("Metrics analyzed:", KEY_METRICS)

############################################
# STATIONARITY TEST
############################################

def test_stationarity(series):
    series = series.dropna()
    
    # Skip if too short
    if len(series) < 20:
        return False, np.nan, np.nan
    
    # Skip if almost constant
    if series.var() < 1e-8:
        return False, np.nan, np.nan

    # Limit length
    if len(series) > MAX_POINTS:
        series = series.iloc[-MAX_POINTS:]

    try:
        adf_p = adfuller(series)[1]
    except Exception:
        adf_p = np.nan
    try:
        kpss_p = kpss(series, regression="c", nlags="auto")[1]
    except Exception:
        kpss_p = np.nan

    stationary = (adf_p < 0.05) and (kpss_p > 0.05)
    return stationary, adf_p, kpss_p

############################################
# ANALYSIS
############################################

results = []

for ahu_id, g in df.groupby("ahu_id"):
    row = {"ahu_id": ahu_id}

    # Only numeric columns
    g_numeric = g[KEY_METRICS].copy()
    g_numeric.index = g["timestamp"]

    # Resample and interpolate
    g_resampled = g_numeric.resample(RESAMPLE_FREQ).mean().interpolate()

    for metric in KEY_METRICS:
        series = g_resampled[metric]
        stationary, adf_p, kpss_p = test_stationarity(series)
        variance = series.var()
        ml_candidate = stationary and variance >= MIN_VARIANCE

        row[f"{metric}_stationary"] = stationary
        row[f"{metric}_variance"] = variance
        row[f"{metric}_ml_candidate"] = ml_candidate
        row[f"{metric}_adf_p"] = adf_p
        row[f"{metric}_kpss_p"] = kpss_p

    results.append(row)

result_df = pd.DataFrame(results)

# Save detailed result
result_df.to_csv(OUTPUT_DETAIL, index=False)
print("Detailed analysis saved to:", OUTPUT_DETAIL)

# Create summary
summary_rows = []
for metric in KEY_METRICS:
    candidates = result_df[f"{metric}_ml_candidate"].sum()
    total = len(result_df)
    summary_rows.append({
        "metric": metric,
        "AHUs_valid_for_ML": candidates,
        "total_AHUs": total,
        "percentage": round(100*candidates/total, 2)
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_SUMMARY, index=False)
print("ML candidate summary saved to:", OUTPUT_SUMMARY)