"""Week 14 - the pieces behind the diagnostic app, with no Streamlit inside.

Run from the kit root, with the venv active and Ollama running:

    python week14/northpeak.py          # trains or loads the model, prints a check

`app_full.py` and `smoke_test.py` import this file. Every function here can be
called from a plain Python session, which is how we test the app without a
browser.

Earlier weeks built the same pieces step by step: the feature table in Week 4,
the random forest in Week 6, the Chroma store in Week 9, the grounded prompt in
Week 10. This file puts them in one place so the app has one import.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import ollama
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA = Path("data")
MANUALS = Path("docs/manuals")
MODEL_PATH = Path("week14/model.joblib")
CHROMA_DIR = "data/chroma"
COLLECTION = "manuals"
CHAT_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"

ROLLING_COLS = ["temperature_c", "vibration_mm_s"]
TYPES = ["pump", "compressor", "conveyor", "chiller"]
MODEL_FEATURES = [
    "load_pct",
    "ambient_c",
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "machine_age",
    "temperature_c_delta30",
    "vibration_mm_s_delta30",
    "temperature_c_trend",
    "vibration_mm_s_trend",
    "type_pump",
    "type_compressor",
    "type_conveyor",
    "type_chiller",
]
TARGET = "fault_next_7d"
TRAIN_BEFORE = "2025-10-01"
SEED = 42

GREEN_MAX = 0.30
ORANGE_MAX = 0.60

RAG_RULES = (
    "Answer only from the context. If the context does not contain the answer, "
    "say: I don't know. Cite the source file in brackets."
)


# ---------------------------------------------------------------- data


def load_tables():
    """The three clean tables. Dates become real datetimes."""
    machines = pd.read_csv(DATA / "clean" / "machines.csv")
    readings = pd.read_csv(DATA / "clean" / "readings.csv", parse_dates=["date"])
    incidents = pd.read_csv(DATA / "clean" / "incidents.csv")
    return machines, readings, incidents


def build_features(readings: pd.DataFrame, machines: pd.DataFrame) -> pd.DataFrame:
    """The Week 4 feature table: merge, age, rolling deltas and trends, one-hot type."""
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

    rolling = [f"{c}_delta30" for c in ROLLING_COLS] + [f"{c}_trend" for c in ROLLING_COLS]
    return df.dropna(subset=rolling).reset_index(drop=True)


# ---------------------------------------------------------------- model


def train_model(features: pd.DataFrame) -> RandomForestClassifier:
    """The Week 6 random forest, trained on the days before 2025-10-01."""
    train = features[features["date"] < TRAIN_BEFORE]
    model = RandomForestClassifier(n_estimators=200, random_state=SEED, class_weight="balanced")
    model.fit(train[MODEL_FEATURES], train[TARGET])
    return model


def train_or_load_model(features: pd.DataFrame) -> RandomForestClassifier:
    """Load week14/model.joblib if it exists, otherwise train it once and save it."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    model = train_model(features)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def predict_last_30(model, features: pd.DataFrame, machine_id: str, as_of=None) -> pd.DataFrame:
    """Fault probability for the 30 days of one machine that end on `as_of` (default: last day)."""
    rows = features[features["machine_id"] == machine_id].sort_values("date")
    if as_of is not None:
        rows = rows[rows["date"] <= pd.Timestamp(as_of)]
    rows = rows.tail(30)
    proba = model.predict_proba(rows[MODEL_FEATURES])[:, 1]
    out = rows[["date", "temperature_c", "vibration_mm_s", "pressure_bar", TARGET]].copy()
    out["probability"] = proba.round(3)
    return out.reset_index(drop=True)


def status(probability: float) -> str:
    """green below 0.30, orange up to 0.60, red above."""
    if probability < GREEN_MAX:
        return "green"
    if probability < ORANGE_MAX:
        return "orange"
    return "red"


# ---------------------------------------------------------------- manuals and RAG


def chunk_paragraphs(text: str, max_chars: int = 600) -> list[str]:
    """Split on blank lines, then merge paragraphs up to max_chars with one paragraph of overlap."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for p in paragraphs:
        if current and len("\n\n".join(current + [p])) > max_chars:
            chunks.append("\n\n".join(current))
            current = [current[-1]]
        current.append(p)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def machine_type_of(source: str, chunk: str) -> str:
    """Tag a chunk with the machine type its manual is about, or 'all'."""
    for t in ("pump", "compressor"):
        if source.startswith(("01-", "02-")) and t in source:
            return t
    if source.startswith("03-"):
        low = chunk.lower()
        if "chiller" in low and "conveyor" not in low:
            return "chiller"
        if "conveyor" in low and "chiller" not in low:
            return "conveyor"
    return "all"


def load_manual_chunks():
    """All chunks of docs/manuals/*.md with their metadata."""
    texts, metas = [], []
    for path in sorted(MANUALS.glob("*.md")):
        for chunk in chunk_paragraphs(path.read_text(encoding="utf-8")):
            texts.append(chunk)
            metas.append({"source": path.name, "machine_type": machine_type_of(path.name, chunk)})
    return texts, metas


def get_vector_store():
    """Open data/chroma; embed the manuals only if the collection is empty."""
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        persist_directory=CHROMA_DIR,
    )
    if store._collection.count() == 0:
        texts, metas = load_manual_chunks()
        store.add_texts(texts, metadatas=metas)
    return store


def top_chunks(store, machine_type: str, question: str, k: int = 3):
    """The k most similar chunks for this machine type; fall back to all manuals."""
    where = {"machine_type": {"$in": [machine_type, "all"]}}
    docs = store.similarity_search(question, k=k, filter=where)
    if len(docs) < k:
        docs = store.similarity_search(f"{machine_type}: {question}", k=k)
    return docs


def explain_question(machine_id: str, machine_type: str, latest: pd.Series, probability: float) -> str:
    return (
        f"Machine {machine_id} is a {machine_type}. Latest reading: temperature "
        f"{latest['temperature_c']:.1f} C, vibration {latest['vibration_mm_s']:.2f} mm/s, "
        f"pressure {latest['pressure_bar']:.2f} bar. The model gives a fault probability of "
        f"{probability:.2f} for the next 7 days. Which normal range, if any, is exceeded, "
        "and what should the technician check first?"
    )


def explain(question: str, docs) -> str:
    """Ask llama3.2 with the retrieved chunks as the only context."""
    context = "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": RAG_RULES},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0},
    )
    return response.message.content


if __name__ == "__main__":
    machines, readings, incidents = load_tables()
    features = build_features(readings, machines)
    print("feature table :", features.shape)
    model = train_or_load_model(features)
    print("model         :", type(model).__name__, "with", model.n_estimators, "trees")
    last = predict_last_30(model, features, "M004", as_of="2025-12-18")
    print(last.tail(6).to_string(index=False))
    p = float(last["probability"].iloc[-1])
    print("probability on 2025-12-18 :", p, "->", status(p))
