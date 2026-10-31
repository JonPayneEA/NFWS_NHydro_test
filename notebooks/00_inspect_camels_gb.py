# Databricks notebook source
# MAGIC %md
# MAGIC # 00 Inspect CAMELS-GB
# MAGIC
# MAGIC Read-only. Writes nothing. Run on any small cluster, no GPU needed.
# MAGIC
# MAGIC It answers four questions before any model is configured:
# MAGIC 1. Which files and attribute columns does the CAMELS-GB Volume hold?
# MAGIC 2. Which FEH22 descriptors are already present (agreed design, Question 16)?
# MAGIC 3. What do the daily time series look like (columns, date range)?
# MAGIC 4. How many basins survive the record-length filter (agreed design, Question 17)?
# MAGIC
# MAGIC Paste the printed output back into the session. Nothing here is a result about hydrology.

# COMMAND ----------

import os
import re
from collections import defaultdict

import pandas as pd

CAMELS_GB_ROOT = (
    "/Volumes/prd_dash_bronze/uk_ceh_environmental_info_data_centre_unrestricted/"
    "camels_gb/format_CSV_camels_gb/LATEST_camels_gb/"
)

# Agreed split (Question 4)
TRAIN = ("1971-10-01", "2000-09-30")
VALID = ("2000-10-01", "2010-09-30")
TEST = ("2010-10-01", "2022-09-30")

# Agreed filter (Question 17): drop on record length only
MIN_TRAIN_YEARS = 10
MIN_TEST_YEARS = 5

# FEH descriptors the portable attribute set (B2) would need. Matched case-insensitively
# against every column name found. Edit the list if the B2 set changes.
FEH_DESCRIPTORS = [
    "area", "altbar", "aspbar", "aspvar", "bfihost", "dplbar", "dpsbar", "farl",
    "ldp", "propwet", "rmed_1h", "rmed_1d", "rmed_2d", "saar", "saar4170",
    "sprhost", "urbconc", "urbext", "urbloc", "wetness",
]

# Column-name fragments that may hold the quality and influence tags for reporting
TAG_HINTS = ["benchmark", "uncert", "qual", "flag", "reservoir", "abs", "meta", "bankfull"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Files in the Volume

# COMMAND ----------

all_files = []
for dirpath, _, filenames in os.walk(CAMELS_GB_ROOT):
    for name in filenames:
        all_files.append(os.path.join(dirpath, name))

print(f"Root: {CAMELS_GB_ROOT}")
print(f"Total files: {len(all_files)}")

by_dir = defaultdict(list)
for path in all_files:
    by_dir[os.path.dirname(path)].append(os.path.basename(path))

for directory, names in sorted(by_dir.items()):
    rel = os.path.relpath(directory, CAMELS_GB_ROOT)
    print(f"\n[{rel}]  {len(names)} files")
    for name in sorted(names)[:8]:
        print(f"   {name}")
    if len(names) > 8:
        print(f"   ... and {len(names) - 8} more")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Attribute files and columns

# COMMAND ----------

attribute_files = sorted(
    p for p in all_files
    if p.lower().endswith(".csv") and "attribute" in os.path.basename(p).lower()
)
print(f"Attribute files found: {len(attribute_files)}")

attribute_columns = {}
for path in attribute_files:
    header = pd.read_csv(path, nrows=0)
    attribute_columns[os.path.basename(path)] = list(header.columns)
    print(f"\n{os.path.basename(path)}  ({len(header.columns)} columns)")
    print("   " + ", ".join(header.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. FEH22 descriptor coverage
# MAGIC
# MAGIC Presence is checked by name only. A name match does not prove the version is FEH22.
# MAGIC Any documentation files are listed so the version can be read from them.

# COMMAND ----------

flat = {}
for fname, cols in attribute_columns.items():
    for col in cols:
        flat.setdefault(col.lower(), []).append(fname)

found = {d: flat[d] for d in FEH_DESCRIPTORS if d in flat}
missing = [d for d in FEH_DESCRIPTORS if d not in flat]

print("FEH descriptors found by name:")
for d, files in found.items():
    print(f"   {d:10s} in {', '.join(files)}")
print(f"\nMissing by exact name ({len(missing)}): {', '.join(missing) if missing else 'none'}")

near = {}
for d in missing:
    hits = [c for c in flat if d in c or c in d]
    if hits:
        near[d] = hits
if near:
    print("\nPossible near matches for the missing names (check these by hand):")
    for d, hits in near.items():
        print(f"   {d:10s} -> {', '.join(hits)}")

docs = [p for p in all_files if re.search(r"(readme|\.txt|\.pdf|\.md|licen[cs]e|metadata)", os.path.basename(p), re.I)]
print("\nDocumentation-like files (read these for the descriptor version):")
for p in sorted(docs):
    print(f"   {os.path.relpath(p, CAMELS_GB_ROOT)}")

print("\nColumns that may hold quality or influence tags:")
for col, files in sorted(flat.items()):
    if any(h in col for h in TAG_HINTS):
        print(f"   {col:35s} in {', '.join(files)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Daily time series

# COMMAND ----------

ts_files = sorted(
    p for p in all_files
    if p.lower().endswith(".csv")
    and "timeseries" in os.path.basename(p).lower()
    and "attribute" not in os.path.basename(p).lower()
)
print(f"Time series files found: {len(ts_files)}")
for p in ts_files[:3]:
    print(f"   {os.path.basename(p)}")

if not ts_files:
    raise RuntimeError("No time series CSVs matched. Paste section 1 back and the pattern will be fixed.")

sample = pd.read_csv(ts_files[0], nrows=5)
print("\nColumns in the first time series file:")
print("   " + ", ".join(sample.columns))
print("\nFirst rows:")
print(sample.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Record-length filter
# MAGIC
# MAGIC Counts valid daily flow values inside the agreed training and test windows for each basin.
# MAGIC A value is valid if it is present and not negative. Years = valid days / 365.25.
# MAGIC The flow column is detected, and printed, so check it is the one you expect.

# COMMAND ----------

def detect_columns(columns):
    cols = list(columns)
    date_col = next((c for c in cols if c.lower() in ("date", "time", "datetime")), cols[0])
    flow = [c for c in cols if "discharge" in c.lower()]
    # Prefer a volumetric flow column, otherwise the first discharge column
    flow_col = next((c for c in flow if "vol" in c.lower()), flow[0] if flow else None)
    return date_col, flow_col


date_col, flow_col = detect_columns(sample.columns)
print(f"Date column: {date_col}")
print(f"Flow column: {flow_col}")
if flow_col is None:
    raise RuntimeError("No discharge column found. Paste section 4 back and the detection will be fixed.")


def basin_id(path):
    """Station number taken from the file name. Check the printed sample below."""
    match = re.search(r"(\d{4,6})", os.path.basename(path))
    return match.group(1) if match else os.path.basename(path)


def valid_years(frame, window):
    start, end = window
    subset = frame.loc[start:end, flow_col]
    return float(((subset.notna()) & (subset >= 0)).sum()) / 365.25


rows = []
for path in ts_files:
    frame = pd.read_csv(path, usecols=[date_col, flow_col], parse_dates=[date_col], index_col=date_col)
    rows.append(
        {
            "basin": basin_id(path),
            "first_date": frame.index.min().date(),
            "last_date": frame.index.max().date(),
            "train_years": valid_years(frame, TRAIN),
            "valid_years": valid_years(frame, VALID),
            "test_years": valid_years(frame, TEST),
        }
    )

records = pd.DataFrame(rows)
print(f"\nSample ids parsed from file names: {records['basin'].head(5).tolist()}")
print(f"Unique ids: {records['basin'].nunique()} of {len(records)} files")

records["keep"] = (records["train_years"] >= MIN_TRAIN_YEARS) & (records["test_years"] >= MIN_TEST_YEARS)

print(f"\nBasins with a time series file: {len(records)}")
print(f"Kept by the record-length filter: {int(records['keep'].sum())}")
print(f"Dropped for training record under {MIN_TRAIN_YEARS} years: {int((records['train_years'] < MIN_TRAIN_YEARS).sum())}")
print(f"Dropped for test record under {MIN_TEST_YEARS} years: {int((records['test_years'] < MIN_TEST_YEARS).sum())}")
print(f"Earliest first date: {records['first_date'].min()}   Latest last date: {records['last_date'].max()}")

print("\nValid years per basin, summary:")
print(records[["train_years", "valid_years", "test_years"]].describe().round(1).to_string())

print("\nDropped basins (id, train years, test years):")
dropped = records.loc[~records["keep"], ["basin", "train_years", "test_years"]].round(1)
print(dropped.to_string(index=False) if len(dropped) else "none")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to paste back
# MAGIC
# MAGIC Sections 1 to 5 as printed. If section 5 is long, paste the counts and the summary table only.
