"""Week 2, Exercise 1 - Explore the raw readings.

Run from the kit root, with the venv active:

    python week02/exercise_1_solution.py

Finds every defect of data/raw/readings_raw.csv without fixing anything:
  1. row count and shape;
  2. info(): which columns have missing values;
  3. isna().sum(): how many missing values per column;
  4. duplicated(): the 150 rows that appear twice;
  5. drop_duplicates(), then count the missing values again;
  6. describe(): impossible minimums and maximums;
  7. the 999.0 temperatures, the negative loads, the vibration spikes;
  8. one defect table, the deliverable.
"""

import pandas as pd

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 12)

# Step 1 - load and count
print("== Step 1: load and count ==")
raw = pd.read_csv("data/raw/readings_raw.csv")
print(raw.shape)
print(raw.head())
print()

# Step 2 - info
print("== Step 2: info ==")
raw.info()
print()

# Step 3 - missing values per column
print("== Step 3: missing values per column ==")
print(raw.isna().sum())
print()

# Step 4 - duplicated rows
print("== Step 4: duplicated rows ==")
print("full duplicates :", raw.duplicated().sum())
print("same reading_id :", raw.duplicated(subset=["reading_id"]).sum())
pair = raw[raw["reading_id"] == 100]
print(pair)
print()

# Step 5 - drop the duplicates, count again
print("== Step 5: drop duplicates, count again ==")
readings = raw.drop_duplicates()
print(readings.shape)
print(readings.isna().sum())
print()

# Step 6 - describe: impossible values
print("== Step 6: describe ==")
cols = ["load_pct", "temperature_c", "vibration_mm_s", "pressure_bar"]
print(readings[cols].describe().round(2))
print()

# Step 7 - count each kind of impossible value
print("== Step 7: impossible values ==")
n_999 = (readings["temperature_c"] == 999.0).sum()
n_neg = (readings["load_pct"] < 0).sum()
n_spike = (readings["vibration_mm_s"] > 30).sum()
print("temperature == 999.0 :", n_999)
print("load_pct < 0         :", n_neg)
print("vibration > 30       :", n_spike)
print(readings.loc[readings["load_pct"] < 0, ["reading_id", "machine_id", "date", "load_pct"]].sort_values("reading_id"))
print()

# Step 8 - the defect table
print("== Step 8: defect table ==")
defects = pd.Series(
    {
        "rows in the raw file": len(raw),
        "duplicated rows": int(raw.duplicated().sum()),
        "rows after drop_duplicates": len(readings),
        "missing temperature_c": int(readings["temperature_c"].isna().sum()),
        "missing vibration_mm_s": int(readings["vibration_mm_s"].isna().sum()),
        "missing pressure_bar": int(readings["pressure_bar"].isna().sum()),
        "temperature_c == 999.0": int(n_999),
        "load_pct < 0": int(n_neg),
        "vibration_mm_s > 30": int(n_spike),
    },
    name="count",
)
print(defects)
