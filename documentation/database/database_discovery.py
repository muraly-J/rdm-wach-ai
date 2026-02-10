import warnings
import urllib3
import os
import pandas as pd
from influxdb_client import InfluxDBClient
from dotenv import load_dotenv
from datetime import datetime

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
try:
    from influxdb_client.client.warnings import MissingPivotFunction
    warnings.filterwarnings("ignore", category=MissingPivotFunction)
except ImportError:
    pass

load_dotenv()

url = os.getenv("url")
token = os.getenv("token")
org = os.getenv("org")
bucket = os.getenv("bucket")

def explore_database():
    client = InfluxDBClient(url=url, token=token, org=org)
    query_api = client.query_api()

    # Flux Queries
    q_measurements = f'import "influxdata/influxdb/schema" schema.measurements(bucket: "{bucket}")'
    q_tags = f'import "influxdata/influxdb/schema" schema.tagKeys(bucket: "{bucket}")'
    q_fields = f'import "influxdata/influxdb/schema" schema.fieldKeys(bucket: "{bucket}")'

    try:
        # Fetching data
        m_df = query_api.query_data_frame(q_measurements)
        t_df = query_api.query_data_frame(q_tags)
        f_df = query_api.query_data_frame(q_fields)

        # 1. GENERATE FILE OUTPUT
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"db_exploration_{bucket}_{timestamp}.txt"
        
        with open(filename, "w") as f:
            f.write(f"--- DATABASE EXPLORATION: {bucket} ---\n")
            f.write(f"Generated on: {datetime.now()}\n\n")
            
            f.write("📍 ALL MEASUREMENTS:\n")
            if not m_df.empty:
                f.write("\n".join([f"- {m}" for m in m_df["_value"].tolist()]))
            
            f.write("\n\n🏷️ TAG KEYS:\n")
            if not t_df.empty:
                tags = [t for t in t_df["_value"].tolist() if not t.startswith("_")]
                f.write(", ".join(tags))

            f.write("\n\n📊 FIELD KEYS:\n")
            if not f_df.empty:
                f.write(", ".join(f_df["_value"].tolist()))

        # 2. GENERATE INTELLIGENT TERMINAL SUMMARY
        print(f"\n✅ Exploration Complete! Full details saved to: {filename}")
        print("-" * 50)
        print(f"📋 SUMMARY FOR BUCKET: {bucket}")
        
        if not m_df.empty:
            m_list = m_df["_value"].tolist()
            # Intelligent grouping (assumes format wach_eXXXX_...)
            devices = sorted(list(set([m.split('_')[1] for m in m_list if len(m.split('_')) > 1])))
            
            print(f"🔹 Measurements: Found {len(m_list)} unique metrics.")
            print(f"🔹 Devices: Detected {len(devices)} devices (from {devices[0]} to {devices[-1]}).")
            print(f"🔹 Metric Types: Power, Volts, Current, Energy, Frequency, etc.")
        
        if not t_df.empty:
            tags = [t for t in t_df["_value"].tolist() if not t.startswith("_")]
            print(f"🔹 Tags: {len(tags)} keys found ({', '.join(tags)}).")

        print("-" * 50)

    except Exception as e:
        print(f"❌ Exploration failed: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    explore_database()