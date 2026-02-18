# extract_energy_data_multi_separate.py
import os
import warnings
import pandas as pd
from influxdb_client import InfluxDBClient
from dotenv import load_dotenv

# --- Silence warnings ---
warnings.filterwarnings("ignore")

load_dotenv()

url = os.getenv("url")
token = os.getenv("token")
org = os.getenv("org")
bucket = os.getenv("bucket")


def fetch_device_data(start_time, end_time, device_id, fields, resample_freq=None):
    """
    Fetch raw pivoted data for a single device from InfluxDB.
    Returns a DataFrame with controller column.
    """
    client = InfluxDBClient(url=url, token=token, org=org, timeout=180000000)
    query_api = client.query_api()

    # Construct regex for measurements
    fields_regex = "|".join(fields)

    flux_query = f'''
    from(bucket: "{bucket}")
      |> range(start: {start_time}, stop: {end_time})
      |> filter(fn: (r) => r._measurement =~ /^wach_{device_id}_({fields_regex})$/)
      |> pivot(rowKey:["_time"], columnKey: ["_measurement"], valueColumn: "_value")
      |> sort(columns: ["_time"])
    '''

    try:
        dfs = query_api.query_data_frame(flux_query)

        # Handle multiple tables
        if isinstance(dfs, list):
            if len(dfs) == 0:
                print(f"No data returned from InfluxDB for device {device_id}")
                return pd.DataFrame()
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = dfs

        # ---- CLEAN DATAFRAME ----
        df = df.drop(
            columns=[c for c in df.columns if c.startswith("_") and c != "_time"],
            errors="ignore",
        )
        df = df.rename(columns={"_time": "time"})
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

        # Convert measurement names → clean field names
        df.columns = df.columns.str.replace(r"^wach_.*?_","", regex=True)

        # Force everything to numeric
        df = df.apply(pd.to_numeric, errors="coerce")

        # ---- RESAMPLING ----
        if resample_freq:
            df = (
                df
                .resample(resample_freq)
                .mean()
                .ffill()
                .fillna(0)
            )

        # ---- ADD CONTROLLER NAME ----
        df["controller"] = device_id

        return df

    except Exception as e:
        print(f"Failed to fetch data for device {device_id}: {e}")
        return pd.DataFrame()

    finally:
        client.close()


# ---------------- EXAMPLE USAGE ----------------
if __name__ == "__main__":

    # List of controllers you want to fetch (change as needed)
    devices = [f"e02{i}" for i in range(10, 12)]  # e0201..e0205

    fields = [
        #"apparent_power_total",
        "current_l1",
        "current_l2",
        "current_l3",
        "power_factor_avg",
        # "power_l1",
        # "power_l2",
        # "power_l3",
        "power_total",
        "volts_l1_n",
        "volts_l2_n",
        "volts_l3_n"
    ]

    start = "2025-01-12T00:00:00Z"
    end   = "2026-02-12T00:00:00Z"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir,"..",".."))
    parquet_folder = os.path.join(project_root,"paraquet_data","raw")
    os.makedirs(parquet_folder, exist_ok=True)

    for device in devices:
        print(f"\nFetching data for {device}...")
        df = fetch_device_data(start, end, device, fields, resample_freq="15T")
        if not df.empty:
            # Reset index so time is a column
            df_reset = df.reset_index()
            # Save to separate parquet file for this controller
            parquet_path = os.path.join(parquet_folder, f"raw_{device}.parquet")
            df_reset.to_parquet(parquet_path, engine="pyarrow")
            print(f"Saved data for {device} to {parquet_path}")
        else:
            print(f"No data for {device}. Skipping save.")
