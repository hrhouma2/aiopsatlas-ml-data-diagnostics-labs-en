"""Week 14, Exercise 1 - One machine, one chart, its incidents.

Run from the kit root, with the venv active:

    streamlit run week14/app.py

The page shows a machine selector, a Plotly line chart of temperature and
vibration over 2025, and the table of incidents of that machine.
The three helper functions have no Streamlit inside, so a test can import
and call them without a browser.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

SIGNALS = ["temperature_c", "vibration_mm_s"]


@st.cache_data
def load_data():
    """Read the three clean CSV files once. Streamlit keeps the result in memory."""
    machines = pd.read_csv("data/clean/machines.csv")
    readings = pd.read_csv("data/clean/readings.csv", parse_dates=["date"])
    incidents = pd.read_csv("data/clean/incidents.csv")
    return machines, readings, incidents


def readings_for(readings: pd.DataFrame, machine_id: str) -> pd.DataFrame:
    """The 365 rows of one machine, sorted by date."""
    rows = readings[readings["machine_id"] == machine_id]
    return rows.sort_values("date").reset_index(drop=True)


def incidents_for(incidents: pd.DataFrame, machine_id: str) -> pd.DataFrame:
    """The incidents of one machine, oldest first, without the technician name."""
    rows = incidents[incidents["machine_id"] == machine_id].sort_values("date")
    cols = ["incident_id", "date", "category", "severity", "downtime_hours", "repair_cost_cad"]
    return rows[cols].reset_index(drop=True)


def make_chart(rows: pd.DataFrame, machine_id: str):
    """One line per signal, each in its own panel so the scales do not fight."""
    long = rows.melt(id_vars="date", value_vars=SIGNALS, var_name="signal", value_name="value")
    fig = px.line(
        long,
        x="date",
        y="value",
        color="signal",
        facet_row="signal",
        title=f"{machine_id} - temperature and vibration, 2025",
        height=500,
    )
    fig.update_yaxes(matches=None, title_text="")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


def main() -> None:
    st.set_page_config(page_title="NorthPeak diagnostics", layout="wide")
    st.title("NorthPeak diagnostics")

    machines, readings, incidents = load_data()
    machine_id = st.selectbox("Machine", machines["machine_id"].tolist())
    machine = machines[machines["machine_id"] == machine_id].iloc[0]
    st.caption(f"{machine['machine_type']} - {machine['site']} - installed {machine['install_year']}")

    rows = readings_for(readings, machine_id)
    st.plotly_chart(make_chart(rows, machine_id), width="stretch")

    st.subheader("Incidents in 2025")
    table = incidents_for(incidents, machine_id)
    st.write(f"{len(table)} incidents")
    st.dataframe(table, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
