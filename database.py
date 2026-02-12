from influxdb_client import InfluxDBClient
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
url = "http://178.128.53.199:8086"
token = "EM3EoRh9djJlo7VYQ_0_G7WLrUJBN2y49Z3AYVzE6LlHVw5kzkHjYdCQL0jKBIgy8kRCW7qiQllYcZCIbqZWtA=="
org = "wach"
bucket = "wach_bucket_3"

# Limit parameters
MAX_MEASUREMENTS = 5    # Max measurements to display
MAX_TAGS = 3            # Max tags to display per measurement
MAX_TAG_VALUES = 5      # Max unique tag values per tag
MAX_FIELDS = 5          # Max fields to display
MAX_SAMPLE_ROWS = 5     # Max sample data points

# -----------------------------
# Connect to InfluxDB
# -----------------------------
client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()

# -----------------------------
# Get all measurements
# -----------------------------
measurements_query = f'''
import "influxdata/influxdb/schema"
schema.measurements(bucket: "{bucket}")
'''
measurements = query_api.query(measurements_query)
measurement_names = [record.get_value() for table in measurements for record in table.records]

print(f"\nFound measurements: {measurement_names[:MAX_MEASUREMENTS]} (showing first {MAX_MEASUREMENTS})\n")

# -----------------------------
# Explore each measurement (limited)
# -----------------------------
for meas in measurement_names[:MAX_MEASUREMENTS]:
    print(f"\n=== Measurement: {meas} ===\n")
    
    # Get Tag Keys
    tag_keys_query = f'''
    import "influxdata/influxdb/schema"
    schema.tagKeys(bucket:"{bucket}", predicate: (r) => r._measurement=="{meas}")
    '''
    tag_keys_res = query_api.query(tag_keys_query)
    tag_keys = [r.get_value() for table in tag_keys_res for r in table.records]
    print(f"Tag Keys (showing first {MAX_TAGS}): {tag_keys[:MAX_TAGS]}")
    
    # Get unique values for each tag (limited)
    for tag in tag_keys[:MAX_TAGS]:
        unique_tag_query = f'''
        from(bucket: "{bucket}")
          |> range(start: -30d)
          |> filter(fn: (r) => r._measurement == "{meas}")
          |> keep(columns: ["{tag}"])
          |> distinct(column: "{tag}")
        '''
        unique_tag_res = query_api.query(unique_tag_query)
        unique_values = [r.get_value() for table in unique_tag_res for r in table.records][:MAX_TAG_VALUES]
        print(f"  - {tag} unique values (first {MAX_TAG_VALUES}): {unique_values}")
    
    # Get Field Keys
    field_keys_query = f'''
    import "influxdata/influxdb/schema"
    schema.fieldKeys(bucket:"{bucket}", predicate: (r) => r._measurement=="{meas}")
    '''
    field_keys_res = query_api.query(field_keys_query)
    field_keys = [(r.get_value(), r.values.get('fieldType')) for table in field_keys_res for r in table.records][:MAX_FIELDS]
    print(f"Field Keys (name, type, showing first {MAX_FIELDS}): {field_keys}")
    
    # Sample Data (last 5 points)
    sample_query = f'''
    from(bucket: "{bucket}")
      |> range(start: -1h)
      |> filter(fn: (r) => r._measurement == "{meas}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> limit(n:{MAX_SAMPLE_ROWS})
    '''
    sample_res = query_api.query(sample_query)
    rows = []
    for table in sample_res:
        for record in table.records:
            rows.append(record.values)
    
    if rows:
        df = pd.DataFrame(rows)
        print(f"\nSample Data (first {MAX_SAMPLE_ROWS} rows):\n{df}\n")
    else:
        print("No data available for this measurement.\n")
