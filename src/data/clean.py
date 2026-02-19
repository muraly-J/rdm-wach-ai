# clean_data_quality_parquet_only.py

import os
import pandas as pd
import numpy as np

# ---------------- SETUP PATHS ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

raw_folder = os.path.join(project_root, "paraquet_data", "raw")
quality_folder = os.path.join(project_root, "paraquet_data", "cleaned")
os.makedirs(quality_folder, exist_ok=True)

# ---------------- FIND RAW FILES ----------------
raw_files = [f for f in os.listdir(raw_folder) if f.endswith(".parquet")]
if not raw_files:
    print("No raw parquet files found in:", raw_folder)
    exit()

print("Found raw files:", raw_files)

# ---------------- DATA QUALITY PARAMETERS ----------------
ranges = {
    'power_total': (0, 10.0),
    'current_l1': (0, 20.0),
    'current_l2': (0, 20.0),
    'current_l3': (0, 20.0),
    'volts_l1_n': (200, 260),
    'volts_l2_n': (200, 260),
    'volts_l3_n': (200, 260),
    'power_factor_avg': (0, 1.0),
}

flatline_metrics = list(ranges.keys())

# ---------------- HELPER FUNCTION ----------------
def detect_sensor_stuck(series, window=8, precision=6):
    """
    Detect true constant sequences.
    Returns boolean mask.
    """
    rounded = series.round(precision)
    change = rounded != rounded.shift()
    group = change.cumsum()
    group_sizes = group.value_counts()
    valid_groups = group_sizes[group_sizes >= window].index
    mask = group.isin(valid_groups)
    return mask

# ---------------- PROCESS EACH CONTROLLER ----------------
for raw_file in raw_files:
    controller_name = raw_file.replace("raw_", "").replace(".parquet", "")
    parquet_path = os.path.join(raw_folder, raw_file)

    print(f"\nProcessing {controller_name}")

    df = pd.read_parquet(parquet_path)
    total_rows = len(df)

    # ---------------- OUT-OF-RANGE CHECK ----------------
    invalid_counts = {}
    for col, (min_val, max_val) in ranges.items():
        if col in df.columns:
            invalid_mask = (df[col] < min_val) | (df[col] > max_val)
            invalid_counts[col] = int(invalid_mask.sum())

    # ---------------- SENSOR STUCK CHECK ----------------
    stuck_counts = {}
    for col in flatline_metrics:
        if col in df.columns:
            mask = detect_sensor_stuck(df[col], window=8, precision=6)
            stuck_counts[col] = int(mask.sum())
        else:
            stuck_counts[col] = 0

    # ---------------- SUMMARY ----------------
    summary_df = pd.DataFrame([{
        "controller": controller_name,
        "total_rows": total_rows,
        "out_of_range_counts": invalid_counts,
        "total_stuck_points": stuck_counts
    }])

    # ---------------- SAVE RESULTS ----------------
    summary_path = os.path.join(
        quality_folder,
        f"data_quality_summary_{controller_name}.csv"
    )

    raw_copy_path = os.path.join(
        quality_folder,
        f"raw_data_{controller_name}.parquet"
    )

    # Save summary as CSV
    summary_df.to_csv(summary_path, index=False)

    # Save raw data as PARQUET
    df.to_parquet(raw_copy_path, index=False, engine="pyarrow")

    print(f"Saved summary → {summary_path}")
    print(f"Saved raw copy (parquet) → {raw_copy_path}")

print("\nAll controllers processed successfully!")
