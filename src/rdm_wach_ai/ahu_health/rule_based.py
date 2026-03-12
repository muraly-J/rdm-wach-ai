import numpy as np
import pandas as pd
from pathlib import Path

############################################
# AUTO PATH DETECTION
############################################

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = PROJECT_ROOT / "data" / "merged" / "ahu_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "meta" / "ahu_health_scores.parquet"
SUMMARY_PATH = PROJECT_ROOT / "data" / "meta" / "diversity_summary.csv"
AHU_SUMMARY_PATH = PROJECT_ROOT / "data" / "meta" / "ahu_health_by_ahu.csv"
HEATMAP_PATH = PROJECT_ROOT / "data" / "meta" / "ahu_failure_heatmap.csv"
TOP10_REPORT_PATH = PROJECT_ROOT / "data" / "meta" / "ahu_top10_report.csv"

print("Project root:", PROJECT_ROOT)
print("Loading data from:", DATA_PATH)

############################################
# CHECK FILE EXISTS
############################################

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

############################################
# CREATE OUTPUT FOLDER
############################################

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

############################################
# LOAD DATA
############################################

df = pd.read_parquet(DATA_PATH)

# Rename and parse timestamp
df = df.rename(columns={"time": "timestamp"})
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

df = df.sort_values(["ahu_id", "timestamp"])

print("Total rows:", len(df))
print("Total AHUs:", df["ahu_id"].nunique())

############################################
# ENERGY DELTA
############################################

df["delta_kwh"] = df.groupby("ahu_id")["energy_import"].diff().fillna(0)

############################################
# CURRENT UNBALANCE
############################################

if "current_unbalance" not in df.columns:
    if {"current_l1", "current_l2", "current_l3"}.issubset(df.columns):
        currents = df[["current_l1", "current_l2", "current_l3"]]
        df["current_unbalance"] = (currents.max(axis=1) - currents.min(axis=1)) / currents.mean(axis=1)

############################################
# THD COMPUTATION (24H rolling)
############################################

thd_cols = [c for c in df.columns if "thd" in c.lower()]
if len(thd_cols) >= 2:
    thd_matrix = df[thd_cols]
    df["composite_thd"] = np.sqrt((thd_matrix**2).mean(axis=1))

    # Rolling 24H window based on timestamp
    df = df.set_index("timestamp")
    df["composite_thd_24h"] = (
        df.groupby("ahu_id")["composite_thd"]
        .rolling("24h", min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df = df.reset_index()

############################################
# CONFIG
############################################

LEVEL_WEIGHT = 0.7
TREND_WEIGHT = 0.3
SLOPE_SENS = 3.0
TREND_WINDOW = 168

SENSITIVITY = {
    "energy": 2.0,
    "pf": 2.5,
    "unbalance": 2.0,
    "thd": 2.0
}

MIN_RSTD = {
    "delta_kwh": 0.05,
    "pf": 0.008,
    "unbalance": 0.15,
    "thd": 0.15,
    "power": 0.05
}

############################################
# FUNCTIONS
############################################

def robust_params(series, min_rstd):
    series = series.dropna()
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    rstd = max(1.4826 * mad, min_rstd)
    return median, rstd

def sigmoid_score(x):
    #s = 1 / (1 + np.exp(-x))
    x=np.clip(x,-50,50)                 #this prevents overflow
    s=1 / (1+np.exp(-x))
    return np.clip(s*2 - 1, 0, 1)

def compute_slope(series):
    y = np.array(series)
    if len(y) < 5:
        return 0
    x = np.arange(len(y))
    return np.polyfit(x, y, 1)[0]

############################################
# SCORING
############################################

results = []

for ahu, g in df.groupby("ahu_id"):
    g = g.sort_values("timestamp")

    med_delta, rstd_delta = robust_params(g["delta_kwh"], MIN_RSTD["delta_kwh"])
    med_pf, rstd_pf = robust_params(g["power_factor_avg"], MIN_RSTD["pf"])
    med_unb, rstd_unb = robust_params(g["current_unbalance"], MIN_RSTD["unbalance"])
    med_thd, rstd_thd = robust_params(g["composite_thd_24h"], MIN_RSTD["thd"])
    med_power, rstd_power = robust_params(g["power_total"], MIN_RSTD["power"])

    p95_power = np.percentile(g["power_total"].dropna(), 95)
    slope_power = compute_slope(g["power_total"].tail(TREND_WINDOW)) / max(rstd_power, 1e-6)

    for _, row in g.iterrows():
        z = (row["delta_kwh"] - med_delta)/rstd_delta
        energy = sigmoid_score((0.6*abs(z)+0.4*max(0,z))*SENSITIVITY["energy"])

        z = (med_pf - row["power_factor_avg"])/rstd_pf
        pf = sigmoid_score(z*SENSITIVITY["pf"])

        z = (row["current_unbalance"] - med_unb)/rstd_unb
        unb = sigmoid_score(z*SENSITIVITY["unbalance"])

        z = (row["composite_thd_24h"] - med_thd)/rstd_thd
        thd = sigmoid_score(z*SENSITIVITY["thd"])

        ratio = row["power_total"]/p95_power if p95_power>0 else 0
        demand = max(0, ratio-0.85)
        scoreA = sigmoid_score(demand*8)
        scoreB = sigmoid_score((row["power_total"]-med_power)/rstd_power*1.5)
        scoreC = sigmoid_score(max(0,slope_power)*SLOPE_SENS)
        overload = 0.5*scoreA + 0.3*scoreB + 0.2*scoreC

        results.append([energy, pf, unb, thd, overload])

############################################
# MERGE RESULTS
############################################

scores = pd.DataFrame(results, columns=[
    "energy_score", "pf_score", "unbalance_score", "thd_score", "overload_score"
])
df = pd.concat([df.reset_index(drop=True), scores], axis=1)

############################################
# CREATE FLAGS
############################################

for c in scores.columns:
    df[c+"_flag"] = df[c] > 0.5

############################################
# GLOBAL DIVERSITY SUMMARY
############################################

summary = []
total = len(df)

for c in scores.columns:
    unhealthy = df[c+"_flag"].sum()
    pct = unhealthy/total*100
    status = "ML_OK" if pct >=5 else "TOO_RARE"
    summary.append([c, unhealthy, pct, status])

summary_df = pd.DataFrame(summary, columns=["metric","unhealthy_hours","percentage","ml_feasible"])
summary_df.to_csv(SUMMARY_PATH, index=False)
print("Saved global summary:", SUMMARY_PATH)

############################################
# PER-AHU SUMMARY
############################################

ahu_summary = df.groupby("ahu_id")[[c+"_flag" for c in scores.columns]].mean()*100
ahu_summary = ahu_summary.rename(columns={
    "energy_score_flag":"energy_%",
    "pf_score_flag":"pf_%",
    "unbalance_score_flag":"unbalance_%",
    "thd_score_flag":"thd_%",
    "overload_score_flag":"overload_%"
})
ahu_summary.to_csv(AHU_SUMMARY_PATH)
print("Saved AHU summary:", AHU_SUMMARY_PATH)

############################################
# HEATMAP DATA (MONTHLY)
############################################

df["month"] = df["timestamp"].dt.tz_localize(None).dt.to_period("M")
heatmap = df.groupby(["ahu_id", "month"])[[c+"_flag" for c in scores.columns]].mean()
heatmap.to_csv(HEATMAP_PATH)
print("Saved heatmap data:", HEATMAP_PATH)

############################################
# SAVE MAIN RESULTS
############################################

df.to_parquet(OUTPUT_PATH)
print("Saved full results:", OUTPUT_PATH)

############################################
# TOP 10 WORST AHU REPORT
############################################

top10 = ahu_summary.mean(axis=1).sort_values(ascending=False).head(10)
top10_report = ahu_summary.loc[top10.index]
top10_report.to_csv(TOP10_REPORT_PATH)
print("Saved Top 10 worst AHUs report:", TOP10_REPORT_PATH)