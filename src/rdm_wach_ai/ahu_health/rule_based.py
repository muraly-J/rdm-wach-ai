import numpy as np
import pandas as pd

############################################
# LOAD DATA
############################################

df = pd.read_parquet(r"D:\AHU\rdm-wach-ai\data\merged\ahu_all.parquet")

# rename time column
df = df.rename(columns={"time": "timestamp"})

# convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# sort
df = df.sort_values(["ahu_id", "timestamp"])

print("Total rows:", len(df))
print("Total AHUs:", df["ahu_id"].nunique())

############################################
# CHECK COLUMNS
############################################

print("\n====== DATASET COLUMNS ======\n")
print(df.columns)

############################################
# CREATE REQUIRED FEATURES
############################################

# ENERGY DELTA
df["delta_kwh"] = df.groupby("ahu_id")["energy_import"].diff()
df["delta_kwh"] = df["delta_kwh"].fillna(0)

############################################
# CURRENT UNBALANCE
############################################

if "current_unbalance" not in df.columns:

    if {"current_l1","current_l2","current_l3"}.issubset(df.columns):

        currents = df[["current_l1","current_l2","current_l3"]]

        df["current_unbalance"] = (
            currents.max(axis=1) - currents.min(axis=1)
        ) / currents.mean(axis=1)

        print("\nCurrent unbalance computed from phase currents")

    else:

        df["current_unbalance"] = 0
        print("\nWARNING: No phase currents found → unbalance set to 0")

############################################
# THD COMPUTATION
############################################

if "composite_thd_24h" not in df.columns:

    thd_cols = [
        c for c in df.columns
        if "thd" in c.lower()
    ]

    print("\nTHD columns detected:", thd_cols)

    if len(thd_cols) >= 3:

        thd_matrix = df[thd_cols]

        # RMS composite THD
        df["composite_thd_24h"] = np.sqrt(
            (thd_matrix**2).mean(axis=1)
        )

        print("Composite THD computed from phase THD")

    else:

        df["composite_thd_24h"] = 0
        print("WARNING: No THD columns → THD set to 0")

############################################
# CONFIG
############################################

LEVEL_WEIGHT = 0.7
TREND_WEIGHT = 0.3
SLOPE_SENS = 3.0

SENSITIVITY = {
    "energy":2.0,
    "pf":2.5,
    "unbalance":2.0,
    "thd":2.0
}

MIN_RSTD = {
    "delta_kwh":0.05,
    "pf":0.008,
    "unbalance":0.15,
    "thd":0.15,
    "power":0.05
}

############################################
# FUNCTIONS
############################################

def robust_params(series, min_rstd):

    series = series.dropna()

    median = np.median(series)

    mad = np.median(np.abs(series - median))

    rstd = 1.4826 * mad

    rstd = max(rstd, min_rstd)

    return median, rstd


def sigmoid_score(x):

    s = 1 / (1 + np.exp(-x))

    return np.clip(s*2 - 1,0,1)


def compute_slope(series):

    y = np.array(series)

    if len(y) < 5:
        return 0

    x = np.arange(len(y))

    slope = np.polyfit(x,y,1)[0]

    return slope

############################################
# RESULT STORAGE
############################################

results = []

############################################
# PROCESS PER AHU
############################################

for ahu, g in df.groupby("ahu_id"):

    g = g.sort_values("timestamp")

    med_delta,rstd_delta = robust_params(g["delta_kwh"],MIN_RSTD["delta_kwh"])
    med_pf,rstd_pf = robust_params(g["power_factor_avg"],MIN_RSTD["pf"])
    med_unb,rstd_unb = robust_params(g["current_unbalance"],MIN_RSTD["unbalance"])
    med_thd,rstd_thd = robust_params(g["composite_thd_24h"],MIN_RSTD["thd"])

    med_power,rstd_power = robust_params(g["power_total"],MIN_RSTD["power"])

    p95_power = np.percentile(g["power_total"].dropna(),95)

    hist_delta = g["delta_kwh"].values
    hist_pf = g["power_factor_avg"].values
    hist_unb = g["current_unbalance"].values
    hist_thd = g["composite_thd_24h"].values
    hist_power = g["power_total"].values

    slope_delta = compute_slope(hist_delta)/rstd_delta
    slope_pf = compute_slope(hist_pf)/rstd_pf
    slope_unb = compute_slope(hist_unb)/rstd_unb
    slope_thd = compute_slope(hist_thd)/rstd_thd
    slope_power = compute_slope(hist_power)/rstd_power

    for i,row in g.iterrows():

        z = (row["delta_kwh"]-med_delta)/rstd_delta
        raw = 0.6*abs(z) + 0.4*max(0,z)

        level = sigmoid_score(raw*SENSITIVITY["energy"])
        trend = sigmoid_score(max(0,slope_delta)*SLOPE_SENS)

        energy_score = LEVEL_WEIGHT*level + TREND_WEIGHT*trend

        z = (med_pf-row["power_factor_avg"])/rstd_pf
        level = sigmoid_score(z*SENSITIVITY["pf"])
        trend = sigmoid_score(max(0,-slope_pf)*SLOPE_SENS)

        pf_score = LEVEL_WEIGHT*level + TREND_WEIGHT*trend

        z = (row["current_unbalance"]-med_unb)/rstd_unb
        level = sigmoid_score(z*SENSITIVITY["unbalance"])
        trend = sigmoid_score(max(0,slope_unb)*SLOPE_SENS)

        unb_score = LEVEL_WEIGHT*level + TREND_WEIGHT*trend

        z = (row["composite_thd_24h"]-med_thd)/rstd_thd
        level = sigmoid_score(z*SENSITIVITY["thd"])
        trend = sigmoid_score(max(0,slope_thd)*SLOPE_SENS)

        thd_score = LEVEL_WEIGHT*level + TREND_WEIGHT*trend

        ratio = row["power_total"]/p95_power
        demand = max(0,ratio-0.85)

        scoreA = sigmoid_score(demand*8)

        z = (row["power_total"]-med_power)/rstd_power
        scoreB = sigmoid_score(z*1.5)

        scoreC = sigmoid_score(max(0,slope_power)*SLOPE_SENS)

        overload_score = 0.5*scoreA + 0.3*scoreB + 0.2*scoreC

        results.append([
            energy_score,
            pf_score,
            unb_score,
            thd_score,
            overload_score
        ])

############################################
# MERGE RESULTS
############################################

scores = pd.DataFrame(results,columns=[
    "energy_score",
    "pf_score",
    "unbalance_score",
    "thd_score",
    "overload_score"
])

df = pd.concat([df.reset_index(drop=True),scores],axis=1)

############################################
# FLAGS
############################################

for c in scores.columns:
    df[c+"_flag"] = df[c] > 0.5

############################################
# DIVERSITY CHECK
############################################

print("\n====== AHU HEALTH DIVERSITY CHECK ======\n")

total = len(df)

for c in scores.columns:

    unhealthy = df[c+"_flag"].sum()

    pct = unhealthy/total*100

    status = "SUFFICIENT" if pct >=5 else "TOO RARE"

    print(c)
    print("unhealthy hours:",unhealthy)
    print("percentage:",round(pct,2),"%")
    print("ML feasibility:",status)
    print()

############################################
# SAVE RESULTS
############################################

output_path = r"D:\AHU\rdm-wach-ai\data\meta\ahu_health_scores.parquet"

df.to_parquet(output_path)

print("\nHealth scores saved to:")
print(output_path)