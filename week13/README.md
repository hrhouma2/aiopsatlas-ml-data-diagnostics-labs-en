# Week 13 — Observability and model reliability: solution scripts

Run every script from the kit root, with the venv active. Exercises 1 and 2 need Ollama running (`llama3.2` and `nomic-embed-text` pulled). Exercise 3 needs only pandas.

| Script | What it does |
| --- | --- |
| `diag_tools.py` | Same two tools as `week12/diag_tools.py`: `query_database(sql)` and `search_manuals(question)`. Copied here so that the week stands alone. |
| `exercise_1_solution.py` | Exercise 1 — Trace: runs the Week 12 agent with `graph.stream(..., stream_mode="updates")` and writes one JSON line per node to `week13/trace.jsonl` (timestamp, node, tool, duration in ms, prompt and output tokens, tokens per second). Reads the trace back with pandas. |
| `exercise_2_solution.py` | Exercise 2 — Golden set: evaluates the RAG on the 10 questions of `week13/golden.json`. Hit rate of the expected manual in the top 3 and keyword presence in the answer. Writes `week13/eval_results.csv`. |
| `exercise_3_solution.py` | Exercise 3 — Drift: Q1 (Jan-Mar) versus Q4 (Oct-Dec) of `data/clean/readings.csv`, mean, std and a 10-bin PSI per sensor. Writes `week13/drift_report.csv`. `ambient_c` drifts by design (seasons); vibration does not. |
| `golden.json` | The 10 golden questions: `question`, `expected_source`, `keywords` (the answer must contain at least one). |

Files created by the scripts: `week13/trace.jsonl`, `week13/eval_results.csv`, `week13/drift_report.csv`. They are rebuilt at every run.

LLM answers and durations change from one machine to another. The scripts check sources, keywords and counts, never an exact sentence.
