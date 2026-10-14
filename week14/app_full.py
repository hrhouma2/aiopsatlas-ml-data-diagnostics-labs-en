"""Week 14, Exercise 2 - The full diagnostic page: data, model, LLM explanation.

Run from the kit root, with the venv active and Ollama running:

    streamlit run week14/app_full.py

Adds to Exercise 1: the fault probability of the last 30 days from the saved
random forest (week14/model.joblib, trained on first run), a coloured status,
and an "Explain" button that asks llama3.2 with the three best manual chunks.
All the logic lives in week14/northpeak.py; this file only draws the page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import northpeak as npk  # noqa: E402

COLOURS = {"green": "#2e8b57", "orange": "#e69500", "red": "#c62828"}
LABELS = {"green": "Normal", "orange": "Watch", "red": "Act now"}


@st.cache_data
def load_everything():
    machines, readings, incidents = npk.load_tables()
    features = npk.build_features(readings, machines)
    return machines, readings, incidents, features


@st.cache_resource
def load_model(features):
    return npk.train_or_load_model(features)


@st.cache_resource
def load_store():
    return npk.get_vector_store()


def probability_chart(window: pd.DataFrame, machine_id: str):
    fig = px.line(window, x="date", y="probability", markers=True,
                  title=f"{machine_id} - fault probability, last 30 days", height=320)
    fig.add_hline(y=npk.GREEN_MAX, line_dash="dot", line_color=COLOURS["orange"])
    fig.add_hline(y=npk.ORANGE_MAX, line_dash="dot", line_color=COLOURS["red"])
    fig.update_yaxes(range=[0, 1], title_text="")
    return fig


def main() -> None:
    st.set_page_config(page_title="NorthPeak diagnostics", layout="wide")
    st.title("NorthPeak diagnostics")

    machines, readings, incidents, features = load_everything()
    model = load_model(features)

    left, right = st.columns([1, 1])
    machine_id = left.selectbox("Machine", machines["machine_id"].tolist())
    dates = features["date"].dt.date.unique()
    as_of = right.select_slider("As of", options=dates, value=dates[-1])

    machine = machines[machines["machine_id"] == machine_id].iloc[0]
    machine_type = machine["machine_type"]
    st.caption(f"{machine_type} - {machine['site']} - installed {machine['install_year']}")

    window = npk.predict_last_30(model, features, machine_id, as_of=as_of)
    latest = window.iloc[-1]
    probability = float(latest["probability"])
    colour = npk.status(probability)

    st.markdown(
        f"<div style='padding:12px;border-radius:8px;background:{COLOURS[colour]};color:white;"
        f"font-size:1.2em'><b>{LABELS[colour]}</b> - fault probability {probability:.2f} "
        f"on {latest['date'].date()}</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(probability_chart(window, machine_id), width="stretch")

    with st.expander("Last 30 days of readings"):
        st.dataframe(window, hide_index=True, width="stretch")

    st.subheader("Why?")
    if st.button("Explain"):
        question = npk.explain_question(machine_id, machine_type, latest, probability)
        with st.spinner("Reading the manuals and asking llama3.2..."):
            docs = npk.top_chunks(load_store(), machine_type, question)
            answer = npk.explain(question, docs)
        st.write(answer)
        with st.expander("Manual chunks used"):
            for d in docs:
                st.markdown(f"**[{d.metadata['source']}]**")
                st.text(d.page_content)
        st.caption("The model writes words, not numbers. Check every range against the table yourself.")

    st.subheader("Incidents in 2025")
    table = incidents[incidents["machine_id"] == machine_id].sort_values("date")
    cols = ["incident_id", "date", "category", "severity", "downtime_hours", "repair_cost_cad"]
    st.dataframe(table[cols], hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
