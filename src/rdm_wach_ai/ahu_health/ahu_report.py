import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

############################################
# PATHS
############################################

PROJECT_ROOT = Path.cwd()

DIVERSITY_PATH = PROJECT_ROOT / "data/meta/diversity_summary.csv"
GO_NO_GO_PATH = PROJECT_ROOT / "data/meta/go_no_go_assessment.csv"
STATIONARITY_PATH = PROJECT_ROOT / "data/meta/metric_ml_summary.csv"

REPORT_FOLDER = PROJECT_ROOT / "data/meta/report"

REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

############################################
# LOAD DATA
############################################

diversity_df = pd.read_csv(DIVERSITY_PATH)
go_df = pd.read_csv(GO_NO_GO_PATH)
station_df = pd.read_csv(STATIONARITY_PATH)

############################################
# MERGE DATA
############################################

report_df = go_df.merge(
    station_df,
    left_on="component",
    right_on="metric",
    how="left"
)

report_df = report_df.drop(columns=["metric"])

############################################
# COMPUTE ML READINESS SCORE
############################################

report_df["anomaly_score"] = report_df["anomaly_percent"] / 100
report_df["ahu_score"] = report_df["ahu_affected_percent"] / 100
report_df["station_score"] = report_df["stationary_percent"] / 100

report_df["ml_readiness_score"] = (
    0.4 * report_df["anomaly_score"] +
    0.3 * report_df["ahu_score"] +
    0.3 * report_df["station_score"]
)

############################################
# SAVE SUMMARY TABLE
############################################

summary_path = REPORT_FOLDER / "ml_feasibility_summary.csv"
report_df.to_csv(summary_path, index=False)

############################################
# PLOT 1: COMPONENT SCORES
############################################

plot_df = report_df[[
    "component",
    "anomaly_percent",
    "ahu_affected_percent",
    "stationary_percent"
]]

plot_df = plot_df.melt(
    id_vars="component",
    var_name="metric",
    value_name="percent"
)

plt.figure(figsize=(10,6))

sns.barplot(
    data=plot_df,
    x="component",
    y="percent",
    hue="metric"
)

plt.title("Component ML Feasibility Metrics")
plt.ylabel("Percent")
plt.xlabel("Component")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(REPORT_FOLDER / "component_scores.png", dpi=300)

plt.close()

############################################
# PLOT 2: ML READINESS
############################################

plt.figure(figsize=(8,5))

sns.barplot(
    data=report_df,
    x="component",
    y="ml_readiness_score"
)

plt.title("ML Readiness Score per Component")

plt.ylabel("Readiness Score")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(REPORT_FOLDER / "ml_readiness_bar.png", dpi=300)

plt.close()

############################################
# GENERATE TEXT REPORT
############################################

report_lines = []

report_lines.append("# ML Feasibility Report\n")

report_lines.append("## Summary\n")

for _, row in report_df.iterrows():

    comp = row["component"]
    decision = row["decision"]

    anomaly = row["anomaly_percent"]
    ahu = row["ahu_affected_percent"]
    stat = row["stationary_percent"]
    score = round(row["ml_readiness_score"],2)

    text = f"""
### {comp}

Decision: **{decision}**

- Anomaly Diversity: {anomaly:.1f}%
- AHU Coverage: {ahu:.1f}%
- Stationary Signals: {stat:.1f}%
- ML Readiness Score: {score}

"""

    report_lines.append(text)

############################################
# FINAL RECOMMENDATION
############################################

go_components = report_df[report_df["decision"]=="GO"]["component"].tolist()
no_components = report_df[report_df["decision"]=="NO-GO"]["component"].tolist()

report_lines.append("## Final Recommendation\n")

report_lines.append(
f"""
ML Ready Components:

{", ".join(go_components)}

Components Requiring More Data:

{", ".join(no_components)}
"""
)

report_text = "\n".join(report_lines)

report_file = REPORT_FOLDER / "ml_feasibility_report.md"

with open(report_file, "w") as f:
    f.write(report_text)

############################################
# DONE
############################################

print("ML Feasibility Report Generated")

print("Summary table:", summary_path)
print("Plots folder:", REPORT_FOLDER)
print("Text report:", report_file)