# feature_engineering.py

import os
import pandas as pd
import numpy as np

# ---------------- FEATURE CREATION FUNCTION ----------------
def create_features(df, target_col="power_total", lags=[1,4,96], rolling_window=4):
    df = df.copy()

    # ---------------- SAFE DATETIME HANDLING ----------------
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.set_index("time")
    else:
        first_col = df.columns[0]
        try:
            df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
            df = df.set_index(first_col)
        except:
            raise ValueError("No datetime column found in dataframe. Add a time/timestamp column.")

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("Index is not datetime. Cannot create time features.")

    # ---------------- TIME FEATURES ----------------
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)

    # ---------------- LAG FEATURES ----------------
    for lag in lags:
        if target_col not in df.columns:
            raise KeyError(f"{target_col} not found in dataframe. Columns: {df.columns.tolist()}")
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    # ---------------- ROLLING FEATURES ----------------
    df[f"{target_col}_rolling_mean_{rolling_window}"] = df[target_col].shift(1).rolling(rolling_window).mean()
    df[f"{target_col}_rolling_std_{rolling_window}"] = df[target_col].shift(1).rolling(rolling_window).std()

    df = df.dropna()
    return df

# ---------------- SAVE FUNCTION ----------------
def save_gold_dataset(df, project_root, filename="ahu_features.parquet"):
    gold_folder = os.path.join(project_root, "paraquet_data", "gold")
    os.makedirs(gold_folder, exist_ok=True)
    path = os.path.join(gold_folder, filename)
    df.to_parquet(path, engine="pyarrow")
    print(f"Saved ML-ready dataset to: {path}")
    return path

# ---------------- MAIN PROCESS ----------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # ✅ Read cleaned parquet raw data, NOT summary CSVs
    clean_folder = os.path.join(project_root, "paraquet_data", "cleaned")
    cleaned_files = [f for f in os.listdir(clean_folder) if f.startswith("raw_data_") and f.endswith(".parquet")]

    if not cleaned_files:
        print("No cleaned raw parquet files found. Run clean.py first.")
        exit()

    all_features = []

    for file in cleaned_files:
        controller_name = file.replace("raw_data_", "").replace(".parquet", "")
        path = os.path.join(clean_folder, file)
        print(f"\nProcessing features for controller: {controller_name} from file: {path}")

        df_clean = pd.read_parquet(path, engine="pyarrow")

        # Create features
        df_feat = create_features(df_clean, target_col="power_total")
        # ---------------- CLEAN FEATURES ----------------
# Drop columns you don't want for ML
        drop_cols = ['result', 'table', 'controller', 'site', 'units']
        df_feat = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])

# Add controller column (for identification only, not a model feature)
        df_feat["controller"] = controller_name

# Debug: see final columns
        print(f"Columns used for ML for {controller_name}: {df_feat.columns.tolist()}")

# Save per-controller dataset
        per_controller_filename = f"features_{controller_name}.parquet"
        save_gold_dataset(df_feat, project_root, per_controller_filename)
        
        # See first 5 rows
        print("First 5 rows of feature dataframe:")
        print(df_feat.head())

# See all columns
        print("\nColumns in dataframe:")
        print(df_feat.columns.tolist())

# See summary of dataframe (types and nulls)
        print("\nDataframe info:")
        print(df_feat.info())

# See statistics for numeric fields
        print("\nDataframe description:")
        print(df_feat.describe())
        
        print(df_feat.columns)

        # Add controller column
        df_feat["controller"] = controller_name

        # Append to list
        all_features.append(df_feat)

        # Save per-controller dataset
        per_controller_filename = f"features_{controller_name}.parquet"
        save_gold_dataset(df_feat, project_root, per_controller_filename)

    # Combine all controllers into single dataset
    if all_features:
        df_gold = pd.concat(all_features, ignore_index=False)
        save_gold_dataset(df_gold, project_root, "ahu_features_all_controllers.parquet")
        print("\nAll controllers combined into a single ML-ready dataset.")