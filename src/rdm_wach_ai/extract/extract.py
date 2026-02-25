from pathlib import Path
import os
import json
import logging
import warnings
import pandas as pd
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from influxdb_client import InfluxDBClient
from influxdb_client.client.warnings import MissingPivotFunction
from dotenv import load_dotenv

# ----------------------------
# Suppress Pivot Warning
# ----------------------------
warnings.simplefilter("ignore", MissingPivotFunction)

# ----------------------------
# Project Root Detection
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CSV_DIR = PROJECT_ROOT / "data" / "csv"
LOG_DIR = PROJECT_ROOT / "logs" / "extraction"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "last_processed.json"
SUMMARY_PATH = PROJECT_ROOT / "data" / "summary.csv"

CONFIG_DIR = PROJECT_ROOT / "src" / "rdm_wach_ai" / "config"
FIELDS_JSON = CONFIG_DIR / "fields.json"
ENV_PATH = CONFIG_DIR / ".env"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ENV_PATH)

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "extraction.log"),
        logging.StreamHandler()
    ]
)

# ----------------------------
# Load Config
# ----------------------------
def load_config():
    with open(FIELDS_JSON, "r") as f:
        fields = json.load(f)["energy_fields"]

    return {
        "url": os.getenv("URL"),
        "token": os.getenv("TOKEN"),
        "org": os.getenv("ORG"),
        "bucket": os.getenv("BUCKET"),
        "fields": fields
    }

# ----------------------------
# Detect Devices
# ----------------------------
def get_all_devices(client, config):
    logging.info("Detecting controllers...")
    query = f'''
    import "influxdata/influxdb/schema"
    schema.measurements(bucket: "{config['bucket']}")
      |> filter(fn: (r) => r._value =~ /^wach_/)
    '''
    df = client.query_api().query_data_frame(query)
    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True)
    if df.empty:
        return []

    measurements = df["_value"].unique()
    devices = set()
    for m in measurements:
        parts = m.split("_")
        if len(parts) >= 2:
            devices.add(parts[1])
    devices = sorted(devices)

    # Print controller names and count
    print(f"\nDetected {len(devices)} controllers:")
    print(", ".join(devices))
    print("="*50)
    return devices

# ----------------------------
# Fetch Monthly Chunk
# ----------------------------
def fetch_chunk(client, device_id, start_str, end_str, config):
    fields_regex = "|".join(config["fields"])
    query = f'''
    from(bucket: "{config['bucket']}")
      |> range(start: {start_str}, stop: {end_str})
      |> filter(fn: (r) => r._measurement =~ /^wach_{device_id}_({fields_regex})$/)
      |> pivot(rowKey:["_time"], columnKey: ["_measurement"], valueColumn: "_value")
    '''
    try:
        df = client.query_api().query_data_frame(query)
        if isinstance(df, list):
            df = pd.concat(df, ignore_index=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.drop(columns=[c for c in df.columns if c.startswith("_") and c != "_time"], errors="ignore")
        df = df.rename(columns={"_time": "time"})
        df.columns = df.columns.str.replace(f"wach_{device_id}_", "", regex=False)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df
    except Exception as e:
        logging.error(f"Error fetching {device_id}: {e}")
        return None

# ----------------------------
# Checkpoint Handling
# ----------------------------
def get_checkpoint(device_id, default_start):
    if CHECKPOINT_PATH.exists():
        try:
            data = json.loads(CHECKPOINT_PATH.read_text())
            return data.get(device_id, default_start)
        except:
            return default_start
    return default_start

def save_checkpoint(device_id, timestamp):
    checkpoints = {}
    if CHECKPOINT_PATH.exists():
        try:
            checkpoints = json.loads(CHECKPOINT_PATH.read_text())
        except:
            checkpoints = {}
    checkpoints[device_id] = timestamp
    CHECKPOINT_PATH.write_text(json.dumps(checkpoints, indent=4))

# ----------------------------
# Summary Update
# ----------------------------
def update_summary(device, final_df):
    summary_data = {}
    if SUMMARY_PATH.exists():
        old_df = pd.read_csv(SUMMARY_PATH)
        for _, row in old_df.iterrows():
            summary_data[row["device"]] = row.to_dict()
    summary_data[device] = {
        "device": device,
        "rows": len(final_df),
        "data_start": final_df["time"].min().isoformat() if not final_df.empty else None,
        "data_end": final_df["time"].max().isoformat() if not final_df.empty else None,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    pd.DataFrame(summary_data.values()).to_csv(SUMMARY_PATH, index=False)

# ----------------------------
# Main Extraction
# ----------------------------
def run_extraction():
    config = load_config()
    client = InfluxDBClient(
        url=config["url"],
        token=config["token"],
        org=config["org"],
        timeout=180_000
    )

    devices = get_all_devices(client, config)
    if not devices:
        print("No controllers found.")
        return

    default_start = "2025-01-12T00:00:00Z"
    final_end_dt = datetime.now(timezone.utc)

    for device in devices:
        print(f"\nProcessing Controller: {device}")

        checkpoint_start = get_checkpoint(device, default_start)
        current_start_dt = pd.to_datetime(checkpoint_start, utc=True)

        parquet_path = RAW_DIR / f"raw_{device}.parquet"
        csv_path = CSV_DIR / f"{device}.csv"

        if parquet_path.exists():
            final_df = pd.read_parquet(parquet_path)
        else:
            final_df = pd.DataFrame()

        while current_start_dt < final_end_dt:
            next_end_dt = min(current_start_dt + relativedelta(months=1), final_end_dt)
            print(f"  → Fetching {current_start_dt.strftime('%Y-%m')}")

            df_chunk = fetch_chunk(
                client,
                device,
                current_start_dt.isoformat(),
                next_end_dt.isoformat(),
                config
            )

            if df_chunk is None:
                break

            if not df_chunk.empty:
                final_df = pd.concat([final_df, df_chunk])
                final_df = final_df.drop_duplicates(subset=["time"])
                final_df.sort_values("time", inplace=True)

                # Save CSV and Parquet
                final_df.to_parquet(parquet_path, engine="pyarrow", index=False)
                final_df.to_csv(csv_path, index=False)

                update_summary(device, final_df)
                print(f"    Added {len(df_chunk)} rows | Total rows: {len(final_df)}")

            save_checkpoint(device, next_end_dt.isoformat())
            current_start_dt = next_end_dt

    client.close()
    print("\nExtraction completed successfully.")

if __name__ == "__main__":
    run_extraction()