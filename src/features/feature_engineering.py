# feature_engineering.py

import os
import pandas as pd
import numpy as np

# ---------------- FEATURE CREATION FUNCTION ----------------
def create_features(df, target_col="power_total", lags=[1,4,96], rolling_window=4):
    """
    Creates features for energy forecasting:
    - Time features: hour, dayofweek, month, is_weekend
    - Lag features
    - Rolling mean/std
    """
    df = df.copy()

    # Ensure time is datetime index
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

    # TIME FEATURES
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)

    # LAG FEATURES
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    # ROLLING FEATURES (shifted by 1 to avoid leakage)
    df[f"{target_col}_rolling_mean_{rolling_window}"] = df[target_col].shift(1).rolling(rolling_window).mean()
    df[f"{target_col}_rolling_std_{rolling_window}"] = df[target_col].shift(1).rolling(rolling_window).std()

    # Drop rows with NaNs created by lag/rolling
    df = df.dropna()
    return df

# ---------------- SAVE FUNCTION ----------------
def save_gold_dataset(df, project_root, filename="ahu_features.parquet"):
    gold_folder = os.path.join(project_root, "paraquet_data", "gold")
    os.makedirs(gold_folder, exist_ok=True)
    
    path = os.path.join(gold_folder, filename)
    df.to_parquet(path, engine="pyarrow")
    print(f"Saved ML-ready dataset to {path}")
    return path

# ---------------- MAIN PROCESS ----------------
if __name__ == "__main__":
    # Project paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    clean_folder = os.path.join(project_root, "paraquet_data", "cleaned")

    # List all cleaned controller files
    cleaned_files = [f for f in os.listdir(clean_folder) if f.endswith(".csv")]
    if not cleaned_files:
        print("No cleaned files found. Run clean.py first.")
        exit()

    all_features = []

    for file in cleaned_files:
        controller_name = file.replace("cleaned_", "").replace(".csv", "")
        path = os.path.join(clean_folder, file)
        print(f"\nProcessing features for controller: {controller_name}")

        df_clean = pd.read_csv(path)

        # Create features
        df_feat = create_features(df_clean, target_col="power_total")

        # Add controller column
        df_feat["controller"] = controller_name

        # Append to list for combined dataset
        all_features.append(df_feat)

        # Optional: save per-controller ML-ready dataset
        per_controller_filename = f"features_{controller_name}.parquet"
        save_gold_dataset(df_feat, project_root, per_controller_filename)

    # Combine all controllers into a single gold dataset
    if all_features:
        df_gold = pd.concat(all_features, ignore_index=False)
        save_gold_dataset(df_gold, project_root, "ahu_features_all_controllers.parquet")
        print("\nAll controllers combined into a single ML-ready dataset.")
