"""Week 2, Exercise 3 - Clean the raw readings and prove it.

Run from the kit root, with the venv active:

    python week02/exercise_3_solution.py

Turns data/raw/readings_raw.csv back into a clean table, one defect at a
time, and checks every step against data/clean/readings.csv with assert:
  1. drop the 150 duplicated rows              -> 14,600 rows
  2. sort by reading_id, same ids as the clean file
  3. negative load_pct -> absolute value        -> equal to the clean file
  4. vibration > 30 -> divide by 25             -> equal to the clean file
  5. temperature 999.0 -> missing               -> 306 missing temperatures
  6. fill missing values with the machine median -> no missing value left
  7. save data/clean/readings_from_raw.csv
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 120)

raw = pd.read_csv("data/raw/readings_raw.csv")
truth = pd.read_csv("data/clean/readings.csv")

# Step 1 - duplicates
print("== Step 1: drop duplicates ==")
readings = raw.drop_duplicates()
print("rows:", len(raw), "->", len(readings))
assert len(readings) == 14600
print()

# Step 2 - same rows as the clean file
print("== Step 2: sort and compare ids ==")
readings = readings.sort_values("reading_id").reset_index(drop=True)
assert (readings["reading_id"] == truth["reading_id"]).all()
assert (readings["ambient_c"] == truth["ambient_c"]).all()
assert (readings["power_kw"] == truth["power_kw"]).all()
print("reading_id, ambient_c and power_kw: identical to the clean file")
print()

# Step 3 - negative loads
print("== Step 3: negative loads ==")
print("load_pct < 0 before:", (readings["load_pct"] < 0).sum())
readings["load_pct"] = readings["load_pct"].abs()
print("load_pct < 0 after :", (readings["load_pct"] < 0).sum())
assert (readings["load_pct"] == truth["load_pct"]).all()
print("load_pct: identical to the clean file")
print()

# Step 4 - vibration spikes
print("== Step 4: vibration spikes ==")
spike = readings["vibration_mm_s"] > 30
print("vibration > 30 before:", spike.sum())
readings.loc[spike, "vibration_mm_s"] = (readings.loc[spike, "vibration_mm_s"] / 25).round(2)
print("vibration > 30 after :", (readings["vibration_mm_s"] > 30).sum())
known = readings["vibration_mm_s"].notna()
assert (readings.loc[known, "vibration_mm_s"] == truth.loc[known, "vibration_mm_s"]).all()
print("vibration_mm_s: identical to the clean file where a value exists")
print()

# Step 5 - the 999.0 placeholder
print("== Step 5: 999.0 becomes missing ==")
print("missing temperature before:", readings["temperature_c"].isna().sum())
readings["temperature_c"] = readings["temperature_c"].replace(999.0, np.nan)
print("missing temperature after :", readings["temperature_c"].isna().sum())
known = readings["temperature_c"].notna()
assert (readings.loc[known, "temperature_c"] == truth.loc[known, "temperature_c"]).all()
print("temperature_c: identical to the clean file where a value exists")
print()

# Step 6 - fill missing values with the median of the same machine
print("== Step 6: fill with the machine median ==")
for col in ["temperature_c", "vibration_mm_s", "pressure_bar"]:
    missing = readings[col].isna()
    median = readings.groupby("machine_id")[col].transform("median")
    readings[col] = readings[col].fillna(median)
    error = (readings.loc[missing, col] - truth.loc[missing, col]).abs().mean()
    print(f"{col:<15} filled {missing.sum():>3} values, mean error vs truth {error:.2f}")
assert readings[["temperature_c", "vibration_mm_s", "pressure_bar"]].isna().sum().sum() == 0
print("no missing value left")
print()

# Step 7 - save
print("== Step 7: save ==")
out = "data/clean/readings_from_raw.csv"
readings.to_csv(out, index=False)
check = pd.read_csv(out)
print(out, check.shape)
assert check.shape == (14600, 9)
print("All checks passed.")
