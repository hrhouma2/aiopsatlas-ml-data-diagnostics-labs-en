# Reproducibility checklist - NorthPeak diagnostic app (Week 14)

Filled on the machine used to write the course. Fill your own copy in Exercise 3:
replace every value with what your machine prints.

## 1. Environment

| Item | Command | Value on the course machine |
| --- | --- | --- |
| Operating system | `python -c "import platform; print(platform.platform())"` | Windows-11 |
| Python | `python --version` | Python 3.12.10 |
| Ollama | `ollama --version` | 0.34.0 |
| Chat model | `ollama list` | `llama3.2:latest` (3.2B, Q4_K_M) |
| Embedding model | `ollama list` | `nomic-embed-text:latest` (137M) |

## 2. Package versions (`pip freeze` excerpt)

```text
chromadb==1.5.9
joblib==1.6.0
langchain==1.4.0
langchain-chroma==1.1.0
langchain-ollama==1.1.0
numpy==2.5.3
ollama==0.6.2
pandas==3.0.5
plotly==7.1.0
scikit-learn==1.9.1
streamlit==1.64.0
```

Save the full list with `pip freeze > week14/requirements-lock.txt` before you hand in a project.

## 3. Seeds and fixed settings

| Where | Setting | Why |
| --- | --- | --- |
| `data/make_dataset.py` | `SEED = 42` | the dataset is byte-identical on every machine |
| `week14/northpeak.py` | `RandomForestClassifier(random_state=42)` | same trees, same probabilities |
| `week14/northpeak.py` | `TRAIN_BEFORE = "2025-10-01"` | the temporal split never moves |
| `week14/northpeak.py` | `options={"temperature": 0}` | the LLM picks the most likely token every time |
| `week14/northpeak.py` | `GREEN_MAX = 0.30`, `ORANGE_MAX = 0.60` | the status colours are a documented decision |

Known limit: the LLM text is stable on one machine but can differ between machines and Ollama versions.
Tests check keywords, never an exact sentence.

## 4. Commands, in order, from the kit root

```text
python data/make_dataset.py          # 40 machines, 14,600 readings, 155 incidents
python week14/northpeak.py           # builds the feature table, trains and saves week14/model.joblib
python week14/smoke_test.py          # 7 checks, exit code 0
python week14/e2e_test.py            # runs both pages without a browser
streamlit run week14/app_full.py     # the app, at http://localhost:8501
```

## 5. Expected values

| Check | Expected |
| --- | --- |
| feature table shape | `(14400, 23)` |
| random forest | 200 trees, AUC 0.942 on the days from 2025-10-01 |
| M004 on 2025-12-18 | probability `0.975`, status `red` |
| Chroma collection `manuals` | 31 chunks |
| `smoke_test.py` | `7/7 checks passed` |

## 6. What is NOT in the repository

- `data/` (generated), `data/chroma/` (generated), `week14/model.joblib` (generated).
- `.streamlit/secrets.toml` (credentials). The app reads them with `st.secrets`, never from the code.
