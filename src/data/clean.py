# clean_energy_data_multi.py

import os
import numpy as np
import pandas as pd

# ---------------- SETUP PATHS ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

raw_folder = os.path.join(project_root, "paraquet_data", "raw")
clean_folder = os.path.join(project_root, "paraquet_data", "cleaned")
os.makedirs(clean_folder, exist_ok=True)

# ---------------- FIND RAW FILES ----------------
raw_files = [f for f in os.listdir(raw_folder) if f.endswith(".parquet")]
if not raw_files:
    print("No raw parquet files found in:", raw_folder)
    exit()

print("Found raw files:", raw_files)

# ---------------- CLEANING PARAMETERS ----------------
# Acceptable value ranges for validation
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

# Columns to drop after cleaning
drop_cols = [
    'result', 'units', 'table', 'site',
    'power_l1', 'power_l2', 'power_l3',
    'apparent_power_total'
]

# ---------------- HELPER FUNCTIONS ----------------
def detect_flatline(series, window=8):
    """Detect flatline segments where rolling std = 0."""
    rolling_std = series.rolling(window=window, min_periods=1).std()
    return rolling_std == 0

# ---------------- PROCESS EACH CONTROLLER ----------------
for raw_file in raw_files:
    controller_name = raw_file.replace("raw_", "").replace(".parquet", "")
    parquet_path = os.path.join(raw_folder, raw_file)

    print(f"\n--- Processing {controller_name} ---")
    df = pd.read_parquet(parquet_path)
    total_rows = len(df)

    # ---------------- RANGE VALIDATION ----------------
    for col, (min_val, max_val) in ranges.items():
        if col in df.columns:
            df.loc[(df[col] < min_val) | (df[col] > max_val), col] = np.nan

    # Save invalid rows for reference
    invalid_rows = df[df[list(ranges.keys())].isna().any(axis=1)]
    invalid_rows.to_csv(
        os.path.join(clean_folder, f"invalid_rows_{controller_name}.csv"), index=False
    )

    # ---------------- INTERPOLATE SINGLE GAPS ----------------
    feature_columns = list(ranges.keys())
    for col in feature_columns:
        is_nan = df[col].isna()
        prev_valid = ~df[col].shift(1).isna()
        next_valid = ~df[col].shift(-1).isna()
        single_gap = is_nan & prev_valid & next_valid
        df.loc[single_gap, col] = df[col].interpolate(method='linear', limit=1)

    # ---------------- DROP BAD ROWS ----------------
    # Drop rows where target (power_total) is missing
    target_missing_rows = df['power_total'].isna().sum()
    df = df[df['power_total'].notna()]

    # Drop rows where >50% features are missing
    nan_count = df[feature_columns].isna().sum(axis=1)
    nan_ratio = nan_count / len(feature_columns)
    sparse_row_count = (nan_ratio > 0.5).sum()
    df = df[nan_ratio <= 0.5].reset_index(drop=True)

    # ---------------- FLATLINE DETECTION ----------------
    flatline_mask = detect_flatline(df['power_total'], window=8)
    flatline_drop_count = flatline_mask.sum()
    df = df[~flatline_mask]

    # ---------------- CLEAN SUMMARY METRICS ----------------
    gold_metrics = {
        'controller': controller_name,
        'total_rows_before_cleaning': total_rows,
        'rows_after_cleaning': len(df),
        'data_retained_percentage': round(len(df) / total_rows * 100, 2),
        'invalid_rows_detected': len(invalid_rows),
        'target_missing_rows_dropped': int(target_missing_rows),
        'sparse_rows_dropped': int(sparse_row_count),
        'flatline_rows_dropped': int(flatline_drop_count),
        'final_missing_values': int(df.isna().sum().sum())
    }

    print(f"Controller {controller_name} cleaning summary:")
    for k, v in gold_metrics.items():
        print(f"{k}: {v}")

    # ---------------- DROP UNWANTED COLUMNS ----------------
    df = df.drop(columns=drop_cols, errors='ignore')

    # ---------------- SAVE CLEANED DATA ----------------
    cleaned_csv_path = os.path.join(clean_folder, f"cleaned_{controller_name}.csv")
    df.to_csv(cleaned_csv_path, index=False)

    # Save cleaning metrics
    metrics_df = pd.DataFrame([gold_metrics])
    metrics_df.to_csv(
        os.path.join(clean_folder, f"cleaning_metrics_{controller_name}.csv"), index=False
    )

    print(f"Saved cleaned data and metrics for {controller_name}")

print("\nAll controllers processed successfully!")
