"""Builds the NorthPeak machine-diagnostics dataset used in every week of the course.

Run from the kit root:

    python data/make_dataset.py

Everything is synthetic and deterministic (seed 42): running the script twice
gives byte-identical files, so every number printed in the lessons can be
checked by anyone.

Outputs
-------
data/raw/      dirty files students clean in Week 2 (missing values, duplicates,
               outliers, inconsistent labels, mixed date formats)
data/clean/    the same data, already clean, used from Week 3 onwards
data/json/     incidents as JSON documents (Week 3, document databases)
data/graph/    nodes and edges of the machine knowledge graph (Week 3 and 9)
data/diagnostics.sqlite   machines, readings and incidents as SQL tables
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parent
YEAR_START = pd.Timestamp("2025-01-01")
DAYS = 365
N_MACHINES = 40

MACHINE_TYPES = ["pump", "compressor", "conveyor", "chiller"]
SITES = ["Montreal", "Quebec City", "Toronto"]
CATEGORIES = ["bearing_wear", "overheating", "leak", "electrical", "sensor_fault"]
TECHNICIANS = ["A. Tremblay", "S. Nguyen", "M. Haddad", "J. Okafor", "L. Fortin"]

# Physical "personality" of each machine type. Power is a clean linear function
# of load so that Week 5 regression works well; temperature and vibration carry
# the fault signal so that Week 5 and 6 classification has something to learn.
TYPE_PARAMS = {
    #            power_kw = a * load + b     temp offset  vib base  pressure base
    "pump":       {"a": 0.42, "b": 3.0, "temp": 18.0, "vib": 2.0, "pressure": 6.0},
    "compressor": {"a": 0.75, "b": 5.0, "temp": 26.0, "vib": 3.5, "pressure": 8.5},
    "conveyor":   {"a": 0.20, "b": 1.5, "temp": 12.0, "vib": 1.2, "pressure": 1.0},
    "chiller":    {"a": 0.60, "b": 4.0, "temp": 8.0,  "vib": 1.6, "pressure": 4.0},
}

DESCRIPTIONS = {
    "bearing_wear": [
        "Loud grinding noise from the {part} bearing. Vibration alarm at {vib:.1f} mm/s.",
        "Bearing temperature rising on {mid}. Operator reports a rattling sound near the {part}.",
        "Vibration trend up for 5 days. Bearing play confirmed on inspection of the {part}.",
    ],
    "overheating": [
        "Casing temperature reached {temp:.0f} C. Unit stopped by the thermal switch.",
        "Overheating alarm on {mid}. Cooling fan running but airflow blocked by dust.",
        "Temperature above limit for 2 hours during peak load. Coolant level low.",
    ],
    "leak": [
        "Oil leak under the {part}. Puddle about 30 cm wide found during the morning round.",
        "Pressure dropped from {pressure:.1f} bar to {low_pressure:.1f} bar. Seal leak suspected.",
        "Coolant leak at the hose fitting near the {part}. Tightened and topped up.",
    ],
    "electrical": [
        "Motor tripped the breaker twice in one shift. Insulation test pending.",
        "Control panel shows fault code E{code}. Contactor replaced.",
        "Intermittent power loss on {mid}. Loose terminal found in the junction box.",
    ],
    "sensor_fault": [
        "Temperature sensor reads {bad:.0f} C, physically impossible. Sensor replaced.",
        "Vibration sensor flatlined at 0.0 for 3 days. Cable damaged by the {part} guard.",
        "Pressure reading frozen at {pressure:.1f} bar while the pump was off. Sensor recalibrated.",
    ],
}
PARTS = ["drive shaft", "motor mount", "outlet flange", "inlet housing", "gearbox"]


def make_machines(rng: np.random.Generator) -> pd.DataFrame:
    ids = [f"M{i:03d}" for i in range(1, N_MACHINES + 1)]
    types = np.repeat(MACHINE_TYPES, N_MACHINES // len(MACHINE_TYPES))
    return pd.DataFrame(
        {
            "machine_id": ids,
            "machine_type": types,
            "site": rng.choice(SITES, size=N_MACHINES, p=[0.5, 0.25, 0.25]),
            "install_year": rng.integers(2012, 2023, size=N_MACHINES),
            "rated_power_kw": [TYPE_PARAMS[t]["a"] * 100 + TYPE_PARAMS[t]["b"] for t in types],
        }
    )


def plan_incidents(rng: np.random.Generator, machines: pd.DataFrame) -> pd.DataFrame:
    """One to nine incidents per machine; old machines break more often."""
    rows = []
    for _, m in machines.iterrows():
        age = 2025 - int(m.install_year)
        n = int(rng.poisson(1.5 + 0.35 * age))
        n = max(1, min(n, 9))
        days = np.sort(rng.choice(np.arange(20, DAYS), size=n, replace=False))
        # Keep incidents at least 12 days apart so the 7-day warning windows never overlap.
        kept = []
        for d in days:
            if not kept or d - kept[-1] >= 12:
                kept.append(int(d))
        for d in kept:
            category = rng.choice(CATEGORIES, p=[0.32, 0.25, 0.18, 0.15, 0.10])
            severity = rng.choice(["low", "medium", "high"], p=[0.45, 0.38, 0.17])
            rows.append((m.machine_id, m.machine_type, d, category, severity))
    return pd.DataFrame(rows, columns=["machine_id", "machine_type", "day", "category", "severity"])


def make_readings(rng: np.random.Generator, machines: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    frames = []
    day_index = np.arange(DAYS)
    # Seasonal ambient temperature, shared by all sites with a small offset.
    season = 8.0 - 14.0 * np.cos(2 * np.pi * (day_index - 15) / 365)
    site_offset = {"Montreal": 0.0, "Quebec City": -2.0, "Toronto": 1.5}
    for _, m in machines.iterrows():
        p = TYPE_PARAMS[m.machine_type]
        age = 2025 - int(m.install_year)
        load = np.clip(rng.normal(62, 18, DAYS), 5, 100)
        # Weekends run lighter.
        weekday = (YEAR_START + pd.to_timedelta(day_index, "D")).weekday
        load = np.where(weekday >= 5, load * 0.55, load).round(1)
        # Each machine has its own baseline, so a model that memorises machines cheats.
        own_temp, own_vib = rng.normal(0, 2.5), rng.normal(0, 0.5)
        ambient = season + site_offset[m.site] + rng.normal(0, 1.5, DAYS)

        fault_signal = np.zeros(DAYS)
        label = np.zeros(DAYS, dtype=int)
        for d in plan.loc[plan.machine_id == m.machine_id, "day"]:
            window = np.arange(max(0, d - 7), d)
            ramp = np.linspace(0.3, 1.0, len(window))
            fault_signal[window] = np.maximum(fault_signal[window], ramp)
            label[window] = 1

        temperature = ambient + 0.35 * load + p["temp"] + 0.4 * age + own_temp + 6.0 * fault_signal + rng.normal(0, 2.2, DAYS)
        vibration = p["vib"] + 0.02 * load + 0.12 * age + own_vib + 2.0 * fault_signal + rng.normal(0, 0.45, DAYS)
        pressure = p["pressure"] + 0.01 * load - 0.5 * fault_signal + rng.normal(0, 0.3, DAYS)
        power = p["a"] * load + p["b"] + 0.05 * age + rng.normal(0, 1.2, DAYS)

        frames.append(
            pd.DataFrame(
                {
                    "machine_id": m.machine_id,
                    "date": (YEAR_START + pd.to_timedelta(day_index, "D")).strftime("%Y-%m-%d"),
                    "load_pct": load,
                    "ambient_c": ambient.round(1),
                    "temperature_c": temperature.round(1),
                    "vibration_mm_s": np.clip(vibration, 0.1, None).round(2),
                    "pressure_bar": np.clip(pressure, 0.2, None).round(2),
                    "power_kw": np.clip(power, 0.5, None).round(1),
                    "fault_next_7d": label,
                }
            )
        )
    readings = pd.concat(frames, ignore_index=True)
    readings.insert(0, "reading_id", np.arange(1, len(readings) + 1))
    return readings


def make_incidents(rng: np.random.Generator, plan: pd.DataFrame, readings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sev_hours = {"low": 3.0, "medium": 9.0, "high": 26.0}
    cat_factor = {"bearing_wear": 1.3, "overheating": 1.0, "leak": 0.9, "electrical": 1.1, "sensor_fault": 0.5}
    for i, r in enumerate(plan.itertuples(index=False), start=1):
        date = YEAR_START + pd.Timedelta(days=int(r.day))
        last = readings[(readings.machine_id == r.machine_id) & (readings.date == date.strftime("%Y-%m-%d"))].iloc[0]
        template = rng.choice(DESCRIPTIONS[r.category])
        text = template.format(
            part=rng.choice(PARTS),
            mid=r.machine_id,
            vib=last.vibration_mm_s,
            temp=last.temperature_c,
            pressure=last.pressure_bar + 1.2,
            low_pressure=last.pressure_bar,
            code=int(rng.integers(11, 99)),
            bad=float(rng.choice([250, 999, -40])),
        )
        downtime = sev_hours[r.severity] * cat_factor[r.category] * float(rng.lognormal(0, 0.25))
        cost = 180 + downtime * 95 + (450 if r.severity == "high" else 0) + float(rng.normal(0, 60))
        rows.append(
            {
                "incident_id": f"INC-{i:04d}",
                "machine_id": r.machine_id,
                "date": date.strftime("%Y-%m-%d"),
                "category": r.category,
                "severity": r.severity,
                "downtime_hours": round(downtime, 1),
                "repair_cost_cad": round(max(cost, 120.0), 2),
                "technician": rng.choice(TECHNICIANS),
                "description": text,
            }
        )
    return pd.DataFrame(rows)


def make_dirty(rng: np.random.Generator, machines: pd.DataFrame, readings: pd.DataFrame, incidents: pd.DataFrame):
    """Week 2 material: every defect here is found again by a specific exercise."""
    m = machines.copy()
    messy = rng.choice(m.index, size=6, replace=False)
    m.loc[messy[:3], "machine_type"] = m.loc[messy[:3], "machine_type"].str.capitalize()
    m.loc[messy[3:], "machine_type"] = " " + m.loc[messy[3:], "machine_type"] + " "
    m.loc[m.index[7], "site"] = "montreal"

    r = readings.drop(columns=["fault_next_7d"]).copy()
    for col, share in [("temperature_c", 0.02), ("vibration_mm_s", 0.015), ("pressure_bar", 0.01)]:
        idx = rng.choice(r.index, size=int(len(r) * share), replace=False)
        r.loc[idx, col] = np.nan
    spikes = rng.choice(r.index, size=25, replace=False)
    r.loc[spikes[:15], "temperature_c"] = 999.0
    r.loc[spikes[15:], "vibration_mm_s"] = (r.loc[spikes[15:], "vibration_mm_s"] * 25).round(2)
    negatives = rng.choice(r.index, size=12, replace=False)
    r.loc[negatives, "load_pct"] = -r.loc[negatives, "load_pct"]
    dupes = r.sample(n=150, random_state=SEED)
    r = pd.concat([r, dupes], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    inc = incidents.copy()
    alt = rng.choice(inc.index, size=int(len(inc) * 0.15), replace=False)
    inc.loc[alt, "date"] = pd.to_datetime(inc.loc[alt, "date"]).dt.strftime("%d/%m/%Y")
    blank = rng.choice(inc.index, size=8, replace=False)
    inc.loc[blank, "technician"] = ""
    return m, r, inc


def write_json(incidents: pd.DataFrame, machines: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    by_machine = machines.set_index("machine_id")
    docs = []
    for row in incidents.to_dict(orient="records"):
        m = by_machine.loc[row["machine_id"]]
        docs.append(
            {
                "_id": row["incident_id"],
                "date": row["date"],
                "machine": {"id": row["machine_id"], "type": m.machine_type, "site": m.site},
                "category": row["category"],
                "severity": row["severity"],
                "downtime_hours": row["downtime_hours"],
                "repair_cost_cad": row["repair_cost_cad"],
                "technician": row["technician"],
                "description": row["description"],
                "tags": sorted({row["category"].split("_")[0], row["severity"], m.machine_type}),
            }
        )
    (out / "incidents.json").write_text(json.dumps(docs, indent=2), encoding="utf-8")


def write_graph(machines: pd.DataFrame, incidents: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    nodes = pd.concat(
        [
            pd.DataFrame({"node_id": machines.machine_id, "label": "Machine", "name": machines.machine_id}),
            pd.DataFrame({"node_id": SITES, "label": "Site", "name": SITES}),
            pd.DataFrame({"node_id": MACHINE_TYPES, "label": "MachineType", "name": MACHINE_TYPES}),
            pd.DataFrame({"node_id": CATEGORIES, "label": "FaultCategory", "name": CATEGORIES}),
            pd.DataFrame({"node_id": incidents.incident_id, "label": "Incident", "name": incidents.incident_id}),
        ],
        ignore_index=True,
    )
    edges = pd.concat(
        [
            pd.DataFrame({"source": machines.machine_id, "relation": "LOCATED_AT", "target": machines.site}),
            pd.DataFrame({"source": machines.machine_id, "relation": "IS_A", "target": machines.machine_type}),
            pd.DataFrame({"source": incidents.incident_id, "relation": "HAPPENED_ON", "target": incidents.machine_id}),
            pd.DataFrame({"source": incidents.incident_id, "relation": "HAS_CATEGORY", "target": incidents.category}),
        ],
        ignore_index=True,
    )
    nodes.to_csv(out / "nodes.csv", index=False)
    edges.to_csv(out / "edges.csv", index=False)


def write_sqlite(machines: pd.DataFrame, readings: pd.DataFrame, incidents: pd.DataFrame, path: Path) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as con:
        machines.to_sql("machines", con, index=False)
        readings.to_sql("readings", con, index=False)
        incidents.to_sql("incidents", con, index=False)
        con.execute("CREATE INDEX idx_readings_machine_date ON readings(machine_id, date)")
        con.execute("CREATE INDEX idx_incidents_machine ON incidents(machine_id)")


def main() -> None:
    rng = np.random.default_rng(SEED)
    machines = make_machines(rng)
    plan = plan_incidents(rng, machines)
    readings = make_readings(rng, machines, plan)
    incidents = make_incidents(rng, plan, readings)

    clean = ROOT / "clean"
    raw = ROOT / "raw"
    clean.mkdir(exist_ok=True)
    raw.mkdir(exist_ok=True)

    machines.to_csv(clean / "machines.csv", index=False)
    readings.to_csv(clean / "readings.csv", index=False)
    incidents.to_csv(clean / "incidents.csv", index=False)

    m_raw, r_raw, i_raw = make_dirty(rng, machines, readings, incidents)
    m_raw.to_csv(raw / "machines_raw.csv", index=False)
    r_raw.to_csv(raw / "readings_raw.csv", index=False)
    i_raw.to_csv(raw / "incidents_raw.csv", index=False)

    write_json(incidents, machines, ROOT / "json")
    write_graph(machines, incidents, ROOT / "graph")
    write_sqlite(machines, readings, incidents, ROOT / "diagnostics.sqlite")

    print(f"machines   : {len(machines):>6} rows  -> data/clean/machines.csv")
    print(f"readings   : {len(readings):>6} rows  -> data/clean/readings.csv  (fault_next_7d = 1 on {readings.fault_next_7d.mean():.1%})")
    print(f"incidents  : {len(incidents):>6} rows  -> data/clean/incidents.csv")
    print(f"raw copies : readings_raw has {len(r_raw)} rows ({len(r_raw) - len(readings)} duplicates), "
          f"{int(r_raw.temperature_c.isna().sum())} missing temperatures")
    print("json/graph/sqlite written.")


if __name__ == "__main__":
    main()
