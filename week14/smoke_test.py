"""Week 14, Exercise 3 - Smoke test of the diagnostic app.

Run from the kit root, with the venv active and Ollama running:

    python week14/smoke_test.py

Seven checks, each printed as PASS or FAIL. The script exits with code 1 if
any check fails, so it can run in a shell script or a CI job. It never opens
a browser: it calls the same functions the app uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import northpeak as npk  # noqa: E402

REQUIRED_FILES = [
    "data/clean/machines.csv",
    "data/clean/readings.csv",
    "data/clean/incidents.csv",
    "data/diagnostics.sqlite",
    "docs/manuals/01-pump-troubleshooting.md",
]

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


# 1. data files exist
missing = [f for f in REQUIRED_FILES if not Path(f).exists()]
check("data files exist", not missing, f"missing: {missing}" if missing else f"{len(REQUIRED_FILES)} files")

# 2. feature table has the expected columns
machines, readings, incidents = npk.load_tables()
features = npk.build_features(readings, machines)
absent = [c for c in npk.MODEL_FEATURES + [npk.TARGET] if c not in features.columns]
check("feature table columns", not absent and len(features) > 0, f"{features.shape}, missing: {absent}")

# 3. model loads
try:
    model = npk.train_or_load_model(features)
    check("model loads", hasattr(model, "predict_proba"), f"{type(model).__name__} from {npk.MODEL_PATH}")
except Exception as exc:  # noqa: BLE001
    model = None
    check("model loads", False, repr(exc))

# 4. a prediction is a probability
if model is not None:
    window = npk.predict_last_30(model, features, "M004", as_of="2025-12-18")
    p = float(window["probability"].iloc[-1])
    check("prediction in [0, 1]", 0.0 <= p <= 1.0, f"M004 on 2025-12-18 -> {p} ({npk.status(p)})")

# 5. Chroma collection is non-empty
try:
    store = npk.get_vector_store()
    n = store._collection.count()
    check("chroma collection non-empty", n > 0, f"{n} chunks in '{npk.COLLECTION}'")
except Exception as exc:  # noqa: BLE001
    store = None
    check("chroma collection non-empty", False, repr(exc))

# 6. retrieval returns chunks for a machine type (needs Ollama for the query embedding)
if store is not None:
    try:
        docs = npk.top_chunks(store, "pump", "vibration rising over several days")
        sources = sorted({d.metadata["source"] for d in docs})
        check("retrieval returns 3 chunks", len(docs) == 3, f"sources: {sources}")
    except Exception as exc:  # noqa: BLE001
        check("retrieval returns 3 chunks", False, repr(exc))

# 7. Ollama answers
try:
    import ollama

    reply = ollama.chat(
        model=npk.CHAT_MODEL,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        options={"temperature": 0},
    ).message.content
    check("ollama answers", "ready" in reply.lower(), f"reply: {reply.strip()[:40]!r}")
except Exception as exc:  # noqa: BLE001
    check("ollama answers", False, repr(exc))

failed = [name for name, ok, _ in results if not ok]
print()
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
