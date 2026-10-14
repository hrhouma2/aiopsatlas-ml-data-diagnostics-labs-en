"""Week 13, Exercise 3 - Drift report: Q1 versus Q4 of readings.csv.

Run from the kit root, with the venv active (no Ollama needed):

    python week13/exercise_3_solution.py

For every sensor column we compare January-March (Q1) with October-December
(Q4): mean, standard deviation and a Population Stability Index (PSI) built
from 10 quantile bins of Q1. Then we check whether the link between the
sensors and the label moved (concept drift) or only the inputs did (data drift).
"""

from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = ["load_pct", "ambient_c", "temperature_c", "vibration_mm_s", "pressure_bar", "power_kw"]
REPORT_PATH = Path("week13/drift_report.csv")


# Step 2 - a simple PSI
def psi(reference, current, bins=10):
    """Population Stability Index. Bins come from the quantiles of `reference`.
    Below 0.1: stable. 0.1 to 0.25: watch. Above 0.25: drift."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_share = np.histogram(current, bins=edges)[0] / len(current)
    ref_share = np.clip(ref_share, 1e-4, None)
    cur_share = np.clip(cur_share, 1e-4, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def verdict(value):
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "watch"
    return "DRIFT"


if __name__ == "__main__":
    # Step 1 - the two quarters
    readings = pd.read_csv("data/clean/readings.csv", parse_dates=["date"])
    q1 = readings[readings["date"].dt.quarter == 1]
    q4 = readings[readings["date"].dt.quarter == 4]
    print("== Step 1: the two quarters ==")
    print(f"Q1 rows: {len(q1)}  ({q1['date'].min().date()} to {q1['date'].max().date()})")
    print(f"Q4 rows: {len(q4)}  ({q4['date'].min().date()} to {q4['date'].max().date()})")

    # Step 3 - mean, std and PSI per sensor
    print("\n== Step 3: drift table ==")
    rows = []
    for column in SENSORS:
        value = psi(q1[column].to_numpy(), q4[column].to_numpy())
        rows.append(
            {
                "sensor": column,
                "q1_mean": round(q1[column].mean(), 2),
                "q4_mean": round(q4[column].mean(), 2),
                "q1_std": round(q1[column].std(), 2),
                "q4_std": round(q4[column].std(), 2),
                "psi": round(value, 3),
                "verdict": verdict(value),
            }
        )
    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    report.to_csv(REPORT_PATH, index=False)

    # Step 4 - where does the temperature drift come from?
    print("\n== Step 4: temperature minus ambient ==")
    for name, part in [("Q1", q1), ("Q4", q4)]:
        gap = (part["temperature_c"] - part["ambient_c"]).mean()
        print(f"{name}: mean temperature_c - ambient_c = {gap:.2f} C")
    print("mean ambient_c by month:")
    print(readings.groupby(readings["date"].dt.month)["ambient_c"].mean().round(1).to_string())

    # Step 5 - data drift or concept drift?
    print("\n== Step 5: did the link with the label move? ==")
    for name, part in [("Q1", q1), ("Q4", q4)]:
        rate = part["fault_next_7d"].mean()
        faulty = part[part["fault_next_7d"] == 1]
        healthy = part[part["fault_next_7d"] == 0]
        vib_gap = faulty["vibration_mm_s"].mean() - healthy["vibration_mm_s"].mean()
        temp_gap = faulty["temperature_c"].mean() - healthy["temperature_c"].mean()
        print(f"{name}: fault rate {rate:.1%} | vibration gap fault-healthy {vib_gap:+.2f} mm/s | temperature gap {temp_gap:+.2f} C")
    print(f"\nsaved {REPORT_PATH}")
