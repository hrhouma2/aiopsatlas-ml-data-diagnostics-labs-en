# Week 1 — AI and LLM foundations: solution scripts

Run every script from the kit root, with the venv active and Ollama running (`ollama pull llama3.2` done).

| Script | What it does |
| --- | --- |
| (no script) | Exercise 1 — Install the kit: clone, venv, `pip install -r requirements.txt`, `python data/make_dataset.py`, first look at the three clean CSVs with pandas. |
| `exercise_2_solution.py` | Exercise 2 — Talk to a local LLM: a general question, then a NorthPeak question the model cannot know, then the same question pushed until the model invents an answer. |
| `exercise_3_solution.py` | Exercise 3 — Prompt vs tool: a hand-made tool call. `count_machines(site)` reads `data/clean/machines.csv`; the model asks for the tool as JSON, our code runs it, the model reads the result. |

LLM answers change from one machine to another. The scripts check keywords and counts, never an exact sentence.
