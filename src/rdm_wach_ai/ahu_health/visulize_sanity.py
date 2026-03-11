import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

############################################
# PATHS
############################################

PROJECT_ROOT = Path.cwd()

DETAIL_PATH = PROJECT_ROOT / "data" / "meta" / "ahu_stationarity_analysis.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "meta" / "metric_ml_summary.csv"

PLOT_FOLDER = PROJECT_ROOT / "data" / "meta" / "plots"

PLOT_FOLDER.mkdir(parents=True, exist_ok=True)

############################################
# LOAD DATA
############################################

detail_df = pd.read_csv(DETAIL_PATH)
summary_df = pd.read_csv(SUMMARY_PATH)

############################################
# 1️⃣ ML CANDIDATE HEATMAP
############################################

ml_cols = [c for c in detail_df.columns if "ml_candidate" in c]

heatmap_df = detail_df.set_index("ahu_id")[ml_cols]

heatmap_df = heatmap_df.astype(int)

plt.figure(figsize=(14,10))

sns.heatmap(
    heatmap_df,
    cmap="RdYlGn",
    cbar=True
)

plt.title("ML Candidate Metrics per AHU")

plt.xlabel("Metrics")
plt.ylabel("AHU")

plt.tight_layout()

plt.savefig(PLOT_FOLDER / "ml_candidate_heatmap.png", dpi=300)

plt.close()

############################################
# 2️⃣ METRIC SUCCESS RATE
############################################

plt.figure(figsize=(8,5))

sns.barplot(
    data=summary_df,
    x="metric",
    y="percentage"
)

plt.title("Percentage of AHUs where Metric is ML Candidate")

plt.ylabel("Percentage of AHUs")

plt.xlabel("Metric")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(PLOT_FOLDER / "metric_success_rate.png", dpi=300)

plt.close()

############################################
# 3️⃣ VARIANCE DISTRIBUTION
############################################

variance_cols = [c for c in detail_df.columns if "variance" in c]

variance_df = detail_df[variance_cols]

plt.figure(figsize=(10,6))

sns.boxplot(data=variance_df)

plt.title("Variance Distribution of AHU Metrics")

plt.ylabel("Variance")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(PLOT_FOLDER / "variance_distribution.png", dpi=300)

plt.close()

############################################
# 4️⃣ STATIONARITY COUNT
############################################

stationary_cols = [c for c in detail_df.columns if "stationary" in c]

stationary_counts = detail_df[stationary_cols].sum()

stationary_counts.index = [c.replace("_stationary","") for c in stationary_counts.index]

plt.figure(figsize=(8,5))

sns.barplot(
    x=stationary_counts.index,
    y=stationary_counts.values
)

plt.title("Number of AHUs with Stationary Signals")

plt.ylabel("Number of AHUs")

plt.xlabel("Metric")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(PLOT_FOLDER / "stationarity_counts.png", dpi=300)

plt.close()

############################################
# DONE
############################################

print("Visualization complete.")
print("Plots saved to:", PLOT_FOLDER)