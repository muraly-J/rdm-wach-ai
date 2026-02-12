# clean_energy_data.py

import pandas as pd
import os

# --- Load raw data ---
raw_file = "data/raw/raw_one_year.csv"
df = pd.read_csv(raw_file)

print(" Raw data loaded")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

# Remove unnecessary metadata columns ---
metadata_cols = ['result', 'table', 'controller', 'site']
columns_to_drop = [col for col in metadata_cols if col in df.columns]
df = df.drop(columns=columns_to_drop)
print(f"Dropped metadata columns: {columns_to_drop}")

# Ensure 'time' is datetime and set as index ---
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
print(" 'time' column set as datetime index")

# Convert all other columns to numeric ---
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

#Check for missing values ---
missing_summary = df.isna().sum()
print("\n--- Missing Values Summary ---")
print(missing_summary[missing_summary > 0])

# Forward-fill then fill remaining with 0
df = df.ffill().fillna(0)
print(" Missing values handled (forward-fill + 0)")

#  Check for any constant or useless columns ---
constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
if constant_cols:
    print(f"Constant columns detected (may be dropped for ML): {constant_cols}")
   
#  Save cleaned data ---
os.makedirs("data/cleaned", exist_ok=True)
clean_file = "data/cleaned/clean_one_year.csv"
df.to_csv(clean_file)
print(f"Cleaned data saved to {clean_file}")

# Quick summary ---
print("\n--- Cleaned Data Summary ---")
print(df.describe())
