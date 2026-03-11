import os
import warnings
import pandas as pd
import re
from influxdb_client import InfluxDBClient
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# -----------------------------
# ENV VARIABLES
# -----------------------------
url = os.getenv("url")
token = os.getenv("token")
org = os.getenv("org")
bucket = os.getenv("bucket")

DATA_FOLDER = "data/raw"
META_FOLDER = "data/meta"

CHECKPOINT_FILE = f"{META_FOLDER}/checkpoint.txt"
STATUS_FILE = f"{META_FOLDER}/status.csv"
CONTROLLER_FILE = f"{META_FOLDER}/controllers.csv"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(META_FOLDER, exist_ok=True)

# -----------------------------
# USER CONTROLLER RANGE
# -----------------------------
START_CONTROLLER = "e0501"
END_CONTROLLER   = "e0511"

# -----------------------------
# CONNECT CLIENT
# -----------------------------
def get_client():
    return InfluxDBClient(
        url=url,
        token=token,
        org=org,
        timeout=600000
    )

# -----------------------------
# DISCOVER CONTROLLERS
# -----------------------------
def discover_controllers():
    print("\nDiscovering controllers from InfluxDB...")
    client = get_client()
    query_api = client.query_api()

    flux = f'''
    import "influxdata/influxdb/schema"
    schema.measurements(bucket: "{bucket}")
    '''

    tables = query_api.query_data_frame(flux)

    if isinstance(tables, list):
        df = pd.concat(tables)
    else:
        df = tables

    measurements = df["_value"].tolist()
    controllers = set()
    for m in measurements:
        match = re.match(r"wach_(.*?)_", m)
        if match:
            controllers.add(match.group(1))

    controllers = sorted(list(controllers))
    pd.DataFrame({"controller": controllers}).to_csv(CONTROLLER_FILE, index=False)

    print(f"\nTotal Controllers Found: {len(controllers)}")
    for c in controllers:
        print(c)

    client.close()
    return controllers

# -----------------------------
# FILTER CONTROLLER RANGE
# -----------------------------
def filter_controller_range(controllers):
    if START_CONTROLLER not in controllers:
        raise ValueError(f"{START_CONTROLLER} not found in controller list")
    if END_CONTROLLER not in controllers:
        raise ValueError(f"{END_CONTROLLER} not found in controller list")
    start_index = controllers.index(START_CONTROLLER)
    end_index = controllers.index(END_CONTROLLER)
    return controllers[start_index:end_index+1]

# -----------------------------
# CHECKPOINT
# -----------------------------
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return f.read().strip()
    return None

def save_checkpoint(controller):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(controller)

# -----------------------------
# SAVE STATUS
# -----------------------------
def save_status(controller, df):
    status = {
        "controller": controller,
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": round(df.memory_usage().sum() / 1e6, 2)
    }
    status_df = pd.DataFrame([status])
    if os.path.exists(STATUS_FILE):
        status_df.to_csv(STATUS_FILE, mode="a", header=False, index=False)
    else:
        status_df.to_csv(STATUS_FILE, index=False)

# -----------------------------
# FETCH DATA
# -----------------------------
def fetch_device_data(start_time, end_time, controller, resample="15T"):
    client = get_client()
    query_api = client.query_api()
    print(f"Querying {controller}")

    flux = f'''
    from(bucket: "{bucket}")
      |> range(start: {start_time}, stop: {end_time})
      |> filter(fn: (r) => r._measurement =~ /^wach_{controller}_.*/)
      |> pivot(rowKey:["_time"], columnKey: ["_measurement"], valueColumn: "_value")
      |> sort(columns: ["_time"])
    '''

    try:
        tables = query_api.query_data_frame(flux)
        if isinstance(tables, list):
            df = pd.concat(tables, ignore_index=True)
        else:
            df = tables

        if df.empty:
            return pd.DataFrame()

        # Keep _time column and clean others
        df = df.rename(columns={"_time": "time"})
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        df = df.drop(columns=[c for c in df.columns if c.startswith("_") and c != "time"], errors="ignore")
        df.columns = [c.replace(f"wach_{controller}_", "") for c in df.columns]
        df = df.apply(pd.to_numeric, errors="coerce")

        # RESAMPLE
        df = df.resample(resample).mean().ffill()

        # ADD CONTROLLER COLUMN
        df["ahu_id"] = controller

        # RESET INDEX so 'time' becomes a column
        df = df.reset_index()

        return df

    except Exception as e:
        print("Error:", e)
        return pd.DataFrame()
    finally:
        client.close()

# -----------------------------
# EXTRACTION PIPELINE
# -----------------------------
def run_pipeline(start_time, end_time):
    controllers = discover_controllers()

    # Filter controller range
    controllers = filter_controller_range(controllers)
    print("\nControllers selected for extraction:")
    print(controllers)

    last = load_checkpoint()
    if last and last in controllers:
        start_index = controllers.index(last) + 1
        controllers = controllers[start_index:]
        print("\nResuming from:", last)

    print("\nControllers remaining:", len(controllers))

    for controller in controllers:
        print("\n----------------------------------")
        print("Processing:", controller)

        df = fetch_device_data(start_time, end_time, controller)
        if df.empty:
            print("No data")
            save_checkpoint(controller)
            continue

        file_path = f"{DATA_FOLDER}/{controller}.parquet"
        df.to_parquet(file_path, index=False)  # 'time' column is now included
        print("Saved:", file_path)
        print("Rows:", len(df), "Columns:", len(df.columns))

        save_status(controller, df)
        save_checkpoint(controller)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    start = "2025-01-12T00:00:00Z"
    end   = "2026-02-12T00:00:00Z"
    run_pipeline(start, end)
    print("\nPipeline finished")