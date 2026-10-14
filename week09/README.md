# Week 9 — Retrieval and vector databases: solution scripts

Run every script from the kit root, with the venv active and Ollama running (`nomic-embed-text` for all three, `llama3.2` for Exercise 3).

| Script | What it does |
| --- | --- |
| `exercise_1_solution.py` | Exercise 1 — split the 6 manuals of `docs/manuals/` on blank lines, merge into chunks of at most about 600 characters with 1 paragraph of overlap, tag `{source, machine_type}`, embed with `nomic-embed-text`, store in the Chroma collection `manuals` under `data/chroma/` (31 vectors). Deletes and rebuilds the collection if it exists. |
| `exercise_2_solution.py` | Exercise 2 — TF-IDF keyword ranking vs Chroma vector ranking for 5 questions, then a hybrid ranking with reciprocal rank fusion. |
| `exercise_3_solution.py` | Exercise 3 — a first RAG answer: retrieve 3 chunks, build a context with `[source: file]` tags, the rules "answer only from the context, say I don't know, cite the source", two answerable questions and one the manuals cannot answer. |

The chunking recipe and the store live in `common/rag.py` (`chunk_manuals()`, `build_or_load_manuals_store()`). Weeks 10 to 14 reuse them; `build_or_load_manuals_store()` rebuilds `data/chroma/` if it is missing or empty.

LLM answers change from one machine to another. The scripts check keywords and citations, never an exact sentence.
