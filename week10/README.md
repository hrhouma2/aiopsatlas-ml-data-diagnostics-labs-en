# Week 10 — Local LLMs and LangChain: solution scripts

Run every script from the kit root, with the venv active and Ollama running (`ollama pull llama3.2` and `ollama pull nomic-embed-text` done).

| Script | What it does |
| --- | --- |
| `exercise_1_solution.py` | Exercise 1 — A chain that summarises an incident: `ChatOllama`, a `ChatPromptTemplate` with six variables, the chain prompt, model and `StrOutputParser()` joined with the pipe operator, one incident, then five with `.batch()`, and a robust check (machine id present, 25 words or fewer). |
| `exercise_2_solution.py` | Exercise 2 — Structured extraction: a Pydantic model `IncidentFacts {category, severity, part}`, `llm.with_structured_output()`, the first 10 incidents of `data/clean/incidents.csv`, category and severity compared with the true columns. Writes `week10/extraction_results.csv`. |
| `exercise_3_solution.py` | Exercise 3 — Retrieval chain over the Chroma store of the manuals (`data/chroma`, collection `manuals`, built in Week 9; rebuilt here by `build_or_load_manuals_store()` if missing). Grounded prompt with the "I don't know" guard, five questions, and a verification of every answer (citation or refusal). |

LLM answers change from one machine to another. The scripts check keywords, counts and JSON fields, never an exact sentence. Every script uses `temperature=0`.
