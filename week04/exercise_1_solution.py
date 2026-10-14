"""Week 4, Exercise 1 - Build the feature table.

Run from the kit root, with the venv active:

    python week04/exercise_1_solution.py

Builds data/features.csv step by step:
  1. merge readings with machines;
  2. sort by machine and date, add machine_age;
  3. per machine: value minus the mean of the previous 30 days (delta30);
  4. per machine: 3-day mean minus 14-day mean (trend);
  5. one-hot machine_type into four 0/1 columns;
  6. drop the first days of each machine, keep 18 columns, save;
  7. prove the new columns carry signal (group means, then a quick AUC),
     and check against common/features.py.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.features import FEATURE_COLUMNS, build_features  # noqa: E402

pd.set_option("display.width", 120)

# Step 1 - merge
print("== Step 1: merge readings with machines ==")
readings = pd.read_csv("data/clean/readings.csv")
machines = pd.read_csv("data/clean/machines.csv")
df = readings.merge(machines, on="machine_id", how="left")
print(readings.shape, machines.shape, "->", df.shape)
print(df[["machine_id", "date", "temperature_c", "machine_type", "install_year"]].head(3))
print()

# Step 2 - sort and machine_age
print("== Step 2: sort, then machine_age ==")
df = df.sort_values(["machine_id", "date"]).reset_index(drop=True)
df["machine_age"] = 2025 - df["install_year"]
print(df.groupby("machine_type")["machine_age"].mean().round(1))
print()

# Step 3 - delta30 per machine
print("== Step 3: value minus the mean of the previous 30 days ==")
grouped = df.groupby("machine_id")
for col in ["temperature_c", "vibration_mm_s"]:
    mean30 = grouped[col].transform(lambda s: s.rolling(30, min_periods=5).mean().shift(1))
    df[f"{col}_delta30"] = df[col] - mean30
m001 = df[df["machine_id"] == "M001"]
print(m001[["date", "temperature_c", "temperature_c_delta30"]].head(7).to_string(index=False))
print()

# Step 4 - trend per machine
print("== Step 4: 3-day mean minus 14-day mean ==")
for col in ["temperature_c", "vibration_mm_s"]:
    mean3 = grouped[col].transform(lambda s: s.rolling(3, min_periods=1).mean())
    mean14 = grouped[col].transform(lambda s: s.rolling(14, min_periods=3).mean())
    df[f"{col}_trend"] = mean3 - mean14
m001 = df[df["machine_id"] == "M001"]
print(m001[["date", "temperature_c", "temperature_c_trend"]].iloc[43:50].to_string(index=False))
print()

# Step 5 - one-hot machine_type
print("== Step 5: one-hot machine_type ==")
for t in ["pump", "compressor", "conveyor", "chiller"]:
    df[f"type_{t}"] = (df["machine_type"] == t).astype(int)
print(df[["machine_type", "type_pump", "type_compressor", "type_conveyor", "type_chiller"]].drop_duplicates().to_string(index=False))
print()

# Step 6 - drop, keep, save
print("== Step 6: drop the first days, keep 18 columns, save ==")
rolling_cols = ["temperature_c_delta30", "vibration_mm_s_delta30", "temperature_c_trend", "vibration_mm_s_trend"]
print("rows with a NaN in a rolling column:", int(df[rolling_cols].isna().any(axis=1).sum()))
features = df.dropna(subset=rolling_cols).reset_index(drop=True)[FEATURE_COLUMNS]
features.to_csv("data/features.csv", index=False)
print("saved data/features.csv", features.shape)
print("first date kept:", features["date"].min())
print("fault_next_7d = 1 on", f"{features['fault_next_7d'].mean():.1%}", "of rows")
print()

# Step 7 - proof
print("== Step 7: do the new columns carry signal? ==")
print(features.groupby("fault_next_7d")[rolling_cols].mean().round(2).to_string())
print()
train = features[features["date"] < "2025-10-01"]
test = features[features["date"] >= "2025-10-01"]
sensors = ["load_pct", "ambient_c", "temperature_c", "vibration_mm_s", "pressure_bar", "power_kw"]
for name, cols in [("sensors only", sensors), ("sensors + rolling", sensors + rolling_cols)]:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(train[cols], train["fault_next_7d"])
    auc = roc_auc_score(test["fault_next_7d"], model.predict_proba(test[cols])[:, 1])
    print(f"{name:18s} AUC {auc:.3f}")
print()
reference = build_features()
pd.testing.assert_frame_equal(features, reference)
print("features.csv matches common.features.build_features(): OK")
