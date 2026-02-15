# feature_engineering.py
import os
import pandas as pd
import numpy as np

def create_features(df, target_col="power_total", lags=[1,4,96], rolling_window=4):
    """
    Creates features for AHU energy forecasting:
    - Time features: hour, dayofweek, month, weekend
    - Lag features
    - Rolling mean/std
    """
    
    df = df.copy()
    print(df.head())
    
    # TIME FEATURES 
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
    
    # LAG FEATURES
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    
    # ROLLING FEATURES (shifted to avoid leakage)
    df[f"rolling_mean_{rolling_window}"] = df[target_col].shift(1).rolling(rolling_window).mean()
    df[f"rolling_std_{rolling_window}"] = df[target_col].shift(1).rolling(rolling_window).std()
    
    # Drop rows with NaNs created by lag/rolling
    df = df.dropna()
    
    return df

def save_gold_dataset(df, project_root, filename="ahu_features.parquet"):
    gold_folder = os.path.join(project_root, "paraquet_data", "gold")
    os.makedirs(gold_folder, exist_ok=True)
    
    path = os.path.join(gold_folder, filename)
    df.to_parquet(path, engine="pyarrow")
    print(f"Saved ML-ready dataset to {path}")
    return path

# Main
if __name__ == "__main__":
    raw_path = "../../paraquet_data/raw/raw_one_year.parquet"
    df_raw = pd.read_parquet(raw_path, engine="pyarrow")
    df_features = create_features(df_raw)

    project_root = r"D:/AHU/rdm-wach-ai/"
    
    save_gold_dataset(df_features, project_root)
