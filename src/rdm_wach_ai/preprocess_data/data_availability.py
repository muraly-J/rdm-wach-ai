import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
MERGED_FOLDER = "data/merged"
RESAMPLE_FREQ = "1D"
OUTPUT_SUMMARY = "data/meta/data_availability_summary.csv"
GRAPH_FOLDER = "data/meta/ahu_graphs"
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# -----------------------------
# FIND MERGED PARQUET FILE
# -----------------------------
parquet_files = [f for f in os.listdir(MERGED_FOLDER) if f.endswith(".parquet")]
if not parquet_files:
    raise FileNotFoundError(f"No parquet file found in {MERGED_FOLDER}")
MERGED_FILE = os.path.join(MERGED_FOLDER, parquet_files[0])
print(f"Using merged Parquet file: {MERGED_FILE}")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_parquet(MERGED_FILE)
df["time"] = pd.to_datetime(df["time"])

# Exclude 'ahu_id' and 'time' from metrics
metrics = [c for c in df.columns if c not in ["ahu_id", "time"]]
ahus = df["ahu_id"].unique()
print(f"Found {len(ahus)} AHUs and {len(metrics)} metrics")

# -----------------------------
# BUILD DAILY DATA AVAILABILITY
# -----------------------------
availability_records = []

for ahu in ahus:
    ahu_df = df[df["ahu_id"] == ahu].copy().set_index("time")
    daily_avail = ahu_df[metrics].resample(RESAMPLE_FREQ).apply(lambda x: x.notna().mean()*100)
    daily_avail["ahu_id"] = ahu
    availability_records.append(daily_avail)

availability_df = pd.concat(availability_records).reset_index()

# -----------------------------
# LINE GRAPH: Daily availability per AHU
# -----------------------------
for ahu in ahus:
    ahu_data = availability_df[availability_df["ahu_id"] == ahu]
    # Average across all metrics per day
    ahu_data["daily_pct_available"] = ahu_data[metrics].mean(axis=1)

    plt.figure(figsize=(15, 5))
    plt.plot(ahu_data["time"], ahu_data["daily_pct_available"], marker='o', linestyle='-')
    plt.title(f"Daily Data Availability (%) for AHU: {ahu}")
    plt.xlabel("Date")
    plt.ylabel("Percent Metrics Available (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    filename = os.path.join(GRAPH_FOLDER, f"ahu_{ahu}_daily_availability.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved line graph: {filename}")

# -----------------------------
# BAR GRAPH: Average availability per metric per AHU
# -----------------------------
summary = df.groupby("ahu_id")[metrics].apply(lambda x: x.notna().mean()*100)

for ahu in ahus:
    ahu_metrics = summary.loc[ahu]
    plt.figure(figsize=(15, 5))
    ahu_metrics.plot(kind='bar')
    plt.title(f"Average Metric Availability (%) for AHU: {ahu}")
    plt.xlabel("Metric")
    plt.ylabel("Percent Available (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    filename = os.path.join(GRAPH_FOLDER, f"ahu_{ahu}_metric_summary.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved bar graph: {filename}")

# -----------------------------
# SAVE SUMMARY CSV
# -----------------------------
os.makedirs(os.path.dirname(OUTPUT_SUMMARY), exist_ok=True)
summary.to_csv(OUTPUT_SUMMARY)
print(f"\nData Availability Summary saved to: {OUTPUT_SUMMARY}")
print(summary.head())