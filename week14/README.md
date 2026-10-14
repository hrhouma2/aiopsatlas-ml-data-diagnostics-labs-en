# Week 14 — Diagnostic application integration: solution scripts

Run everything from the kit root, with the venv active and Ollama running (`llama3.2` and `nomic-embed-text` pulled).

| File | What it does |
| --- | --- |
| `app.py` | Exercise 1 — One machine, one chart, its incidents. `streamlit run week14/app.py`. Machine selector, Plotly chart of temperature and vibration over 2025, incident table. |
| `northpeak.py` | The pieces behind the full app, without Streamlit: feature table (Week 4 recipe), random forest (Week 6 recipe, saved to `model.joblib`), 30-day probabilities, status colours, manual chunks, Chroma store (`data/chroma`, collection `manuals`), grounded `llama3.2` explanation. `python week14/northpeak.py` trains the model on first run. |
| `app_full.py` | Exercise 2 — The full page. `streamlit run week14/app_full.py`. Adds the fault probability of the last 30 days, a green / orange / red status, a date slider and an "Explain" button. |
| `smoke_test.py` | Exercise 3 — Seven checks: data files, feature columns, model, prediction in [0, 1], Chroma non-empty, retrieval, Ollama answers. Exit code 0 when all pass. |
| `e2e_test.py` | Exercise 3 — End-to-end test of both pages with `streamlit.testing.v1.AppTest`, no browser. Selects M004 on 2025-12-18, clicks "Explain", checks a keyword. |
| `REPRODUCIBILITY.md` | Exercise 3 — The filled checklist: versions, seeds, commands in order, expected values. |

Generated on first run, do not commit it: `model.joblib` (about 15 MB).

If Ollama runs but Python says `Failed to connect to Ollama`, check the variable `OLLAMA_HOST`. A value of `0.0.0.0:11434` is fine for the server but not for a client on Windows. Set it to `http://127.0.0.1:11434` in your terminal.
