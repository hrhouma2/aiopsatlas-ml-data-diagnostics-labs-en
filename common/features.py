"""The canonical NorthPeak feature table, built in Week 4 and reused from Week 5 on.

Usage, from the kit root:

    from common.features import build_features
    features = build_features()
    features.to_csv("data/features.csv", index=False)

Or on the command line:

    python common/features.py

Column list and the exact recipe are documented in week04/README.md.
"""

from __future__ import annotations

import pandas as pd

ROLLING_COLS = ["temperature_c", "vibration_mm_s"]
TYPES = ["pump", "compressor", "conveyor", "chiller"]
FEATURE_COLUMNS = [
    "machine_id",
    "date",
    "load_pct",
    "ambient_c",
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "power_kw",
    "machine_age",
    "temperature_c_delta30",
    "vibration_mm_s_delta30",
    "temperature_c_trend",
    "vibration_mm_s_trend",
    "type_pump",
    "type_compressor",
    "type_conveyor",
    "type_chiller",
    "fault_next_7d",
]


def build_features(
    readings_path: str = "data/clean/readings.csv",
    machines_path: str = "data/clean/machines.csv",
) -> pd.DataFrame:
    """Merge readings with machines and add the engineered columns.

    Steps, in this order:
      1. merge on machine_id, sort by machine_id then date;
      2. machine_age = 2025 - install_year;
      3. per machine, for temperature_c and vibration_mm_s:
           <col>_delta30 = value - rolling(30, min_periods=5).mean().shift(1)
           <col>_trend   = rolling(3, min_periods=1).mean() - rolling(14, min_periods=3).mean()
      4. one-hot machine_type into type_pump, type_compressor, type_conveyor, type_chiller (0/1);
      5. drop the rows where a rolling column is NaN (the first days of each machine);
      6. keep FEATURE_COLUMNS, in that order.
    """
    readings = pd.read_csv(readings_path)
    machines = pd.read_csv(machines_path)

    df = readings.merge(machines, on="machine_id", how="left")
    df = df.sort_values(["machine_id", "date"]).reset_index(drop=True)

    df["machine_age"] = 2025 - df["install_year"]

    grouped = df.groupby("machine_id")
    for col in ROLLING_COLS:
        mean30 = grouped[col].transform(lambda s: s.rolling(30, min_periods=5).mean().shift(1))
        mean3 = grouped[col].transform(lambda s: s.rolling(3, min_periods=1).mean())
        mean14 = grouped[col].transform(lambda s: s.rolling(14, min_periods=3).mean())
        df[f"{col}_delta30"] = df[col] - mean30
        df[f"{col}_trend"] = mean3 - mean14

    for t in TYPES:
        df[f"type_{t}"] = (df["machine_type"] == t).astype(int)

    rolling_cols = [f"{c}_delta30" for c in ROLLING_COLS] + [f"{c}_trend" for c in ROLLING_COLS]
    df = df.dropna(subset=rolling_cols).reset_index(drop=True)

    return df[FEATURE_COLUMNS]


if __name__ == "__main__":
    features = build_features()
    features.to_csv("data/features.csv", index=False)
    print(features.shape)
    print(features.head())
