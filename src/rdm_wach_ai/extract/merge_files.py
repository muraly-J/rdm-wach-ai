from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# -----------------------------
# Folders and files
# -----------------------------
RAW_DIR = Path("data/raw")          # folder containing individual controller parquet files
MERGED_DIR = Path("data/merged")    # folder to save the merged parquet
MERGED_DIR.mkdir(parents=True, exist_ok=True)
MERGED_FILE = MERGED_DIR / "ahu_all.parquet"

# -----------------------------
# Get all parquet files
# -----------------------------
files = list(RAW_DIR.glob("*.parquet"))
print(f"Found {len(files)} parquet files to merge:")

tables = []

for f in files:
    print("Loading:", f.name)

    # Read parquet using PyArrow (efficient lazy loading)
    table = pq.read_table(f)
    df = table.to_pandas()

    # Ensure 'time' column exists
    if 'time' not in df.columns:
        df.reset_index(inplace=True)  # sometimes index holds time
        if 'time' not in df.columns:
            raise ValueError(f"'time' column missing in {f.name}")

    # Ensure 'ahu_id' column exists
    if 'ahu_id' not in df.columns:
        # If controller id is missing, infer from filename
        df['ahu_id'] = f.stem

    # Append as PyArrow table
    tables.append(pa.Table.from_pandas(df))

# -----------------------------
# Merge all tables
# -----------------------------
merged_table = pa.concat_tables(tables, promote_options='default')

# Convert to Pandas for sorting
df_merged = merged_table.to_pandas()

# -----------------------------
# Sort by controller and time
# -----------------------------
df_merged["time"] = pd.to_datetime(df_merged["time"])
df_merged = df_merged.sort_values(["ahu_id", "time"]).reset_index(drop=True)

# -----------------------------
# Save merged parquet
# -----------------------------
df_merged.to_parquet(MERGED_FILE, index=False, engine="pyarrow", compression="snappy")

print("Merged parquet saved to:", MERGED_FILE)
print("Shape:", df_merged.shape)
print("Controllers:", df_merged['ahu_id'].unique())