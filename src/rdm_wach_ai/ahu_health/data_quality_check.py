import pandas as pd
import numpy as np
from pathlib import Path

############################################
# PATHS
############################################

PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / "data/merged/ahu_all.parquet"
REPORT_PATH = PROJECT_ROOT / "data/meta/data_quality_report.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data/merged/ahu_clean.parquet"

############################################
# LOAD DATA
############################################

df = pd.read_parquet(DATA_PATH)
print("Original columns:", len(df.columns))

############################################
# REMOVE METADATA
############################################

DROP_COLUMNS = ["result","table","controller","site","digital_input_1_and_2"]
df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
print("Columns after metadata removal:", len(df.columns))

############################################
# DETECT METRICS AUTOMATICALLY
############################################

metrics = df.select_dtypes(include=np.number).columns.tolist()

############################################
# DATA QUALITY ANALYSIS
############################################

results = []

for metric in metrics:
    series = df[metric]

    missing_percent = series.isna().mean() * 100
    completeness_score = 1 - series.isna().mean()
    
    unique_ratio = series.nunique() / len(series)
    
    # Dead sensor detection
    flat_sensor = series.nunique() <= 5
    dead_sensor_penalty = 0 if flat_sensor else 1

    std = series.std()
    median = series.median()
    noise_ratio = std / abs(median) if median != 0 else 0
    noise_score = max(0, 1 - noise_ratio)

    quality_score = 0.4 * completeness_score + 0.3 * unique_ratio * dead_sensor_penalty + 0.3 * noise_score

    results.append({
        "metric": metric,
        "missing_percent": round(missing_percent,2),
        "unique_ratio": round(unique_ratio,3),
        "noise_ratio": round(noise_ratio,3),
        "flat_sensor": flat_sensor,
        "quality_score": round(quality_score,3)
    })

############################################
# SAVE REPORT
############################################

report_df = pd.DataFrame(results)
report_df.to_csv(REPORT_PATH, index=False)
print("Data Quality Report saved:", REPORT_PATH)
print(report_df)

############################################
# REMOVE BAD SENSORS
############################################

QUALITY_THRESHOLD = 0.4
bad_sensors = report_df.loc[report_df["quality_score"] < QUALITY_THRESHOLD, "metric"].tolist()
df_clean = df.drop(columns=bad_sensors, errors="ignore")
print("Columns after removing bad sensors:", len(df_clean.columns))

############################################
# SAVE CLEAN DATASET
############################################

df_clean.to_parquet(CLEAN_DATA_PATH)
print("Clean dataset saved:", CLEAN_DATA_PATH)