# extract_energy_data.py
#resample_freq="15T"   (for 15min data use this )

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


def fetch_device_data(start_time, end_time, device_ids, fields, resample_freq=None):
    """
    Fetches raw pivoted data for given devices and fields from InfluxDB.

    Returns:
        pd.DataFrame: Pivoted and optionally resampled DataFrame
    """

    client = InfluxDBClient(url=url, token=token, org=org,timeout=18000000)
    query_api = client.query_api()

    # Construct regex for measurements
    devices_regex = "|".join(device_ids)
    fields_regex = "|".join(fields)

    flux_query = f'''
    from(bucket: "{bucket}")
      |> range(start: {start_time}, stop: {end_time})
      |> filter(fn: (r) => r._measurement =~ /^wach_({devices_regex})_({fields_regex})$/)
      |> pivot(rowKey:["_time"], columnKey: ["_measurement"], valueColumn: "_value")
      |> sort(columns: ["_time"])
    '''

    try:
        dfs = query_api.query_data_frame(flux_query)

        # Handle multiple tables
        if isinstance(dfs, list):
            if len(dfs) == 0:
                print("No data returned from InfluxDB")
                return pd.DataFrame()
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = dfs

        # ---- CLEAN DATAFRAME ----

        # Drop extra Influx columns
        df = df.drop(
            columns=[c for c in df.columns if c.startswith("_") and c != "_time"],
            errors="ignore",
        )

        # Rename time column and set index
        df = df.rename(columns={"_time": "time"})
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

        # Convert measurement names → clean field names
        df.columns = df.columns.str.replace(r"^wach_.*?_","", regex=True)

        # Force everything to numeric (CRITICAL FIX)
        df = df.apply(pd.to_numeric, errors="coerce")

        # ---- RESAMPLING (CORRECT VERSION) ----
        if resample_freq:
            df = (
                df
                .resample(resample_freq)
                .mean()      # average within each period
                .ffill()     # forward-fill gaps
                .fillna(0)   # final safety fill
            )

        return df

    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return pd.DataFrame()

    finally:
        client.close()


# ---------------- EXAMPLE USAGE ----------------
if __name__ == "__main__":

    devices = ["e0206"]
    
    fields = [
    # "apparent_power_l1",
    # "apparent_power_l2",
    # "apparent_power_l3",
    "apparent_power_total",

    "current_l1",
    "current_l2",
    "current_l3",

    "power_factor_avg",
    # "power_factor_l1",
    # "power_factor_l2",
    # "power_factor_l3",

    "power_l1",
    "power_l2",
    "power_l3",
    "power_total",

    "volts_l1_n",
    "volts_l2_n",
    "volts_l3_n"
]


    start = "2025-01-12T00:00:00Z"
    end   = "2026-02-12T00:00:00Z"

    df = fetch_device_data(
        start, end,
        devices,
        fields,
        resample_freq="15T"
    )

    print("\n===== FINAL DATA =====")
    print(df.head())
    
    


    # os.makedirs("RDM-WACH-AI/paraquet_data/raw", exist_ok=True)
    
    # #save as parquet
    # df.to_parquet("RDM-WACH-AI/paraquet_data/raw/raw_one_year.parquet",engine="pyarrow")
    
    #get absolute path of script
    script_dir=os.path.dirname(os.path.abspath(__file__))
    
    #define project root (two levels up from src/data/)
    project_root=os.path.abspath(os.path.join(script_dir,"..",".."))
    
    #define target folder
    parquet_folder=os.path.join(project_root,"paraquet_data","raw")
    os.makedirs(parquet_folder,exist_ok=True)
    
    #save parquet
    parquet_path=os.path.join(parquet_folder,"raw_one_year.parquet")
    df.to_parquet(parquet_path,engine="pyarrow")
    
    print(f"saved to {parquet_path}")
