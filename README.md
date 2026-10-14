# Machine Learning for Data Diagnostics — lab kit

Hands-on kit of the course **[Machine Learning for Data Diagnostics: From Raw Data to Local LLM Agents](https://aiopsatlas.com/en/cours/machine-learning-for-data-diagnostics)** on aiopsatlas.com.

Everything runs on your laptop, for free, with no API key: Python, SQLite, DuckDB and a local LLM through Ollama. Every command in the course was run on this kit. Every number printed in the lessons comes from the dataset built by `data/make_dataset.py`.

## The story

NorthPeak Manufacturing runs 40 machines on 3 sites in Canada. Sensors send one reading per machine per day. Technicians write short incident notes. Your job, over 15 weeks: turn this raw data into a diagnostic application that predicts faults 7 days ahead and explains them with a local LLM.

| Machines | Sites | Readings (2025) | Incidents | Manuals |
| --- | --- | --- | --- | --- |
| 40 (10 pumps, 10 compressors, 10 conveyors, 10 chillers) | Montreal, Quebec City, Toronto | 14,600 rows | 155 | 6 documents |

## Start in 5 commands

```bash
# Linux, macOS, WSL 2, Git Bash                # Windows PowerShell
git clone https://github.com/hrhouma2/aiopsatlas-ml-data-diagnostics-labs-en.git
cd aiopsatlas-ml-data-diagnostics-labs-en
python -m venv .venv                          py -3.12 -m venv .venv
source .venv/bin/activate                     .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python data/make_dataset.py
```

You need Python 3.10 or newer. The first `pip install` downloads about 1 GB (scikit-learn, Chroma, Streamlit, LangChain). It takes a few minutes.

For Weeks 8 to 14 you also need [Ollama](https://ollama.com/download) and two models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## What is in the kit

| Path | Content |
| --- | --- |
| `data/make_dataset.py` | builds the whole dataset, deterministic (seed 42) |
| `data/clean/` | `machines.csv`, `readings.csv`, `incidents.csv`, ready to use |
| `data/raw/` | the same data with real-life defects, for the cleaning week |
| `data/json/incidents.json` | incidents as nested JSON documents |
| `data/graph/` | nodes and edges of the machine knowledge graph |
| `data/diagnostics.sqlite` | the three clean tables in one SQLite database |
| `docs/manuals/` | troubleshooting guides and policies, the corpus for retrieval and RAG |
| `weekNN/` | solution scripts of the guided exercises of week NN |

The generated files under `data/` are not committed. Run `python data/make_dataset.py` once; it takes about 10 seconds.

## Weekly schedule

| Week | Topic | Folder |
| --- | --- | --- |
| 1 | AI and LLM foundations | `week01/` |
| 2 | Python data retrieval and exploration | `week02/` |
| 3 | Data sources and database types | `week03/` |
| 4 | Data preparation and dataset separation | `week04/` |
| 5 | Regression, classification and evaluation | `week05/` |
| 6 | Model selection and validation | `week06/` |
| 7 | Clustering and neural networks | `week07/` |
| 8 | NLP and text representations | `week08/` |
| 9 | Retrieval and vector databases | `week09/` |
| 10 | Local LLMs and LangChain | `week10/` |
| 11 | LLM tools and database querying | `week11/` |
| 12 | Stateful workflows with LangGraph | `week12/` |
| 13 | Observability and model reliability | `week13/` |
| 14 | Diagnostic application integration | `week14/` |
| 15 | Final project and presentation | `week15/` |

## Versions used to write the course

Python 3.12 · pandas 3.0 · numpy 2.5 · scikit-learn 1.9 · duckdb 1.5 · matplotlib 3.11 · Ollama 0.34 with `llama3.2` (3B) and `nomic-embed-text`. Newer versions should work; if an output differs slightly from the lesson, that is why.

## License

MIT. The data is synthetic: no real company, machine or person.
