import os
import glob
import pandas as pd

os.makedirs("../../paraquet_data/reports/", exist_ok=True)

#files = glob.glob("data/gold/*.parquet")
files = glob.glob("../../paraquet_data/raw/*.parquet")


all_reports = []

for file in files:
    df = pd.read_parquet(file)

    stats = pd.DataFrame({
        "mean": df.mean(numeric_only=True),
        "std": df.std(numeric_only=True),
        "missing_%": df.isna().mean() * 100
    })

    stats = stats.reset_index()
    stats.rename(columns={"index": "column"}, inplace=True)

    stats["dataset"] = os.path.basename(file)

    all_reports.append(stats)


final_report = pd.concat(all_reports)
final_report.to_csv("../../paraquet_data/reports/stats.csv", index=False)

print("Stats report generated.")
