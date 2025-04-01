import sqlite3
import pandas as pd

# === CONFIG ===
DB_PATH = "report1.sqlite"

# === CONNECT ===
conn = sqlite3.connect(DB_PATH)

# === LOAD NVTX EVENTS ===
nvtx_df = pd.read_sql("SELECT text, start, end FROM NVTX_EVENTS", conn)
nvtx_df['duration_ms'] = (nvtx_df['end'] - nvtx_df['start']) / 1e6

# === LOAD CUDA KERNELS ===
kernels_df = pd.read_sql("SELECT name, start, end, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL", conn)
kernels_df['duration_ms'] = (kernels_df['end'] - kernels_df['start']) / 1e6

# === LOAD MEMCPY EVENTS ===
memcpy_df = pd.read_sql("SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY", conn)
memcpy_df['duration_ms'] = (memcpy_df['end'] - memcpy_df['start']) / 1e6
memcpy_df['size_MB'] = memcpy_df['bytes'] / (1024**2)

# === ASSIGN EVENTS TO NVTX RANGES ===
def find_nvtx_range(ts, ranges):
    for _, row in ranges.iterrows():
        if row['start'] <= ts <= row['end']:
            return row['text']
    return "unlabeled"

kernels_df['nvtx_range'] = kernels_df['start'].apply(lambda ts: find_nvtx_range(ts, nvtx_df))
memcpy_df['nvtx_range'] = memcpy_df['start'].apply(lambda ts: find_nvtx_range(ts, nvtx_df))

# === SUMMARY: TOTAL KERNEL TIME PER NVTX RANGE ===
kernel_summary = kernels_df.groupby('nvtx_range')['duration_ms'].sum().reset_index()
kernel_summary = kernel_summary.sort_values(by='duration_ms', ascending=False)

# === SUMMARY: TOTAL MEMCPY TIME AND SIZE PER NVTX RANGE ===
memcpy_summary = memcpy_df.groupby('nvtx_range')[['duration_ms', 'size_MB']].sum().reset_index()
memcpy_summary = memcpy_summary.sort_values(by='duration_ms', ascending=False)

# === DISPLAY RESULTS ===
print("\n🔹 Total CUDA Kernel Time by NVTX Range:")
print(kernel_summary.to_string(index=False))

print("\n🔹 Total Memcpy Time and Size by NVTX Range:")
print(memcpy_summary.to_string(index=False))

# === CLOSE DB ===
conn.close()
