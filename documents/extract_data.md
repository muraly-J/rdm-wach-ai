
**Project:** AHU Fleet Data Extraction System  
**Environment:** Python + InfluxDB + Pandas + Parquet  

---

## Objectives for Today

- Improve InfluxDB extraction monitoring  
- Implement monthly progress tracking  
- Enable incremental `summary.csv` updates  
- Limit controller processing to first 154 devices  
- Fix and suppress InfluxDB pivot warning  
- Improve logging visibility in terminal  
- Reduce query timeout errors  
- Implement chunked extraction for large time ranges  

---

## Work Completed

###  Improved Device Detection
- Queried all available controllers from InfluxDB bucket using a small recent window to prevent heavy full-history scan.  
- Displayed total number of detected controllers in terminal.  
- Confirmed detection of **176 controllers**.  
- Limited processing to **first 154 controllers** for controlled execution.  
- Reusable InfluxDB client implemented with **increased timeout** to avoid `ReadTimeoutError`.  

Example terminal output:
Detecting available devices...
Detected 176 devices.
Processing first 154 devices for extraction.
Devices: ['e0101', 'e0102', ..., 'e0254']


---

###  Monthly Chunk-Based Extraction
- Implemented **monthly extraction chunks** instead of very large queries.  
- Ensures each device is processed in manageable time frames.  
- Added incremental checkpoint updates per device after each month.  
- Real-time progress logging:
  - Device ID being processed  
  - Month/time range being extracted  
  - Chunk row count  
  - Cumulative total rows  

Example terminal output:
Processing Device: e0101
Extraction Window: 2025-01-12T00:00:00Z ---> 2026-02-25T00:00:00Z
Month: 2025-01 | Chunk Rows: 1420 | Cumulative Rows: 1420
Month: 2025-02 | Chunk Rows: 1385 | Cumulative Rows: 2805
...
Saved e0101 | Total Rows: 56789


---

###  Checkpoint and Summary Management
- Implemented **robust checkpoint system** to resume extraction if interrupted.  
- Updated `last_processed.json` after every chunk extraction.  
- Ensured **summary.csv** updates incrementally after each device is processed.  
- Summary now includes:
  - Device ID  
  - Total rows extracted  
  - Start and end timestamp of data  
  - Column information  

Example summary entry:
device,data_start,data_end,rows
e0101,2025-01-12T00:00:00Z,2026-02-25T00:00:00Z,56789


---

###  Optimizations and Fixes
- Increased InfluxDB client timeout to **180 seconds** to prevent timeout on large queries.  
- Reduced pivot query load by chunking in 7-day or monthly intervals.  
- Suppressed irrelevant InfluxDB pivot warnings.  
- Improved logging format for **both terminal and file output**.  
- Ensured timezone-aware timestamps (`UTC`) in all datasets.  
- Saved final data in **Parquet format** for efficient storage and processing.  

---

## Outcome
- Extraction pipeline is now **stable, chunked, and resumable**.  
- Can handle **154 devices** with minimal timeout risk.  
- Checkpoints and summaries ensure **incremental and safe data extraction**.  
- Terminal logging provides **real-time monitoring** and progress visibility.

---

## Next Steps
- Extend extraction to **remaining controllers** in batches.  
- Implement **daily or weekly automated extraction** using cron or task scheduler.  
- Optimize memory usage for **pivot-heavy operations** with large fleets.  
- Add **alert system** for devices with no data or query failures.