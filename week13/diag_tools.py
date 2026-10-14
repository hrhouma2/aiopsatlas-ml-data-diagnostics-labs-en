"""Week 13 - the two tools of the diagnostic agent (same file as week12/diag_tools.py).

Weeks 9 to 11 built these pieces step by step (chunking, Chroma store,
read-only SQL with a validator). This file puts them together so that the
Week 13 scripts can import them:

    from diag_tools import query_database, search_manuals, run_sql, SCHEMA

Run from the kit root. Ollama must be running with `llama3.2` and
`nomic-embed-text` pulled.
"""

import re
import sqlite3
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings

MANUALS_DIR = Path("docs/manuals")
CHROMA_DIR = "data/chroma"
DB_URI = "file:data/diagnostics.sqlite?mode=ro"
MAX_CHUNK_CHARS = 600
ROW_LIMIT = 50

MACHINE_TYPE_OF = {
    "01-pump-troubleshooting.md": "pump",
    "02-compressor-troubleshooting.md": "compressor",
    "03-conveyor-and-chiller-troubleshooting.md": "conveyor,chiller",
}

SCHEMA = """SQLite tables of NorthPeak Manufacturing:
machines(machine_id TEXT, machine_type TEXT, site TEXT, install_year INTEGER, rated_power_kw REAL)
readings(reading_id, machine_id TEXT, date TEXT 'YYYY-MM-DD', load_pct, ambient_c, temperature_c, vibration_mm_s, pressure_bar, power_kw, fault_next_7d)
incidents(incident_id TEXT, machine_id TEXT, date TEXT, category TEXT, severity TEXT, downtime_hours, repair_cost_cad, technician TEXT, description TEXT)
machine_type values: pump, compressor, conveyor, chiller. site values: Montreal, Quebec City, Toronto.
category values: bearing_wear, overheating, leak, electrical, sensor_fault. severity values: low, medium, high."""


# ---------------------------------------------------------------- manuals ---
def chunk_text(text, max_chars=MAX_CHUNK_CHARS):
    """Split on blank lines, then merge paragraphs into chunks of about
    max_chars. Each new chunk starts with the last paragraph of the previous
    one (one paragraph of overlap)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], []
    for paragraph in paragraphs:
        candidate = "\n\n".join(current + [paragraph])
        if current and len(candidate) > max_chars:
            chunks.append("\n\n".join(current))
            current = [current[-1], paragraph]
        else:
            current.append(paragraph)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def get_vector_store():
    """Open the persistent Chroma collection. Embed the 6 manuals only when
    the collection is empty (Week 9 did the same)."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    store = Chroma(
        collection_name="manuals",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    if store.get(limit=1)["ids"]:
        return store
    texts, metadatas, ids = [], [], []
    for path in sorted(MANUALS_DIR.glob("*.md")):
        for i, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
            texts.append(chunk)
            metadatas.append(
                {"source": path.name, "machine_type": MACHINE_TYPE_OF.get(path.name, "all")}
            )
            ids.append(f"{path.name}-{i}")
    store.add_texts(texts, metadatas=metadatas, ids=ids)
    return store


def retrieve(question, k=3):
    """Top-k chunks for a question, as LangChain Documents."""
    return get_vector_store().similarity_search(question, k=k)


# -------------------------------------------------------------------- SQL ---
SELECT_ONLY = re.compile(r"^\s*select\b", re.IGNORECASE)
FORBIDDEN = re.compile(
    r";|\b(insert|update|delete|drop|alter|pragma|attach|create|replace)\b", re.IGNORECASE
)


def run_sql(sql):
    """Run one SELECT on the read-only database. Reject anything else.
    Add LIMIT 50 when the query has no LIMIT. Return the rows as text."""
    if not SELECT_ONLY.match(sql) or FORBIDDEN.search(sql):
        return "REJECTED: only one SELECT statement is allowed."
    if not re.search(r"\blimit\b", sql, re.IGNORECASE):
        sql = sql.rstrip() + f" LIMIT {ROW_LIMIT}"
    con = sqlite3.connect(DB_URI, uri=True)
    try:
        cursor = con.execute(sql)
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchall()
    except sqlite3.Error as error:
        return f"SQL ERROR: {error}"
    finally:
        con.close()
    if not rows:
        return "(no rows)"
    lines = [" | ".join(columns)]
    lines += [" | ".join(str(value) for value in row) for row in rows]
    return "\n".join(lines)


# ------------------------------------------------------------------ tools ---
@tool
def query_database(sql: str) -> str:
    """Run one read-only SQL SELECT on the NorthPeak database.
    Tables: machines, readings, incidents. Returns at most 50 rows as text."""
    return run_sql(sql)


@tool
def search_manuals(question: str) -> str:
    """Search the 6 NorthPeak maintenance manuals (operating ranges, symptoms,
    severity policy, sensor guide). Returns the 3 best passages with [source]."""
    docs = retrieve(question, k=3)
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)


TOOLS = [query_database, search_manuals]

if __name__ == "__main__":
    store = get_vector_store()
    print("chunks in the store:", len(store.get()["ids"]))
    print(run_sql("SELECT site, COUNT(*) AS n FROM machines GROUP BY site"))
    print(run_sql("DROP TABLE incidents"))
    print(search_manuals.invoke({"question": "When should a pump be stopped for vibration?"})[:400])
