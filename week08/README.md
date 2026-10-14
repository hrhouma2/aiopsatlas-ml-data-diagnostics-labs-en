# Week 8 — NLP and text representations: solution scripts

Run every script from the kit root, with the venv active. Exercises 2 and 3 need Ollama running with `nomic-embed-text` and `llama3.2` pulled.

| Script | What it does |
| --- | --- |
| `exercise_1_solution.py` | Exercise 1 — TF-IDF on the 155 incident descriptions, stratified split (`random_state=42`), logistic regression predicting `category`, accuracy and confusion matrix, top words per category. |
| `exercise_2_solution.py` | Exercise 2 — `ollama.embed(model="nomic-embed-text")` on the 155 descriptions, cosine similarity with numpy, the 3 nearest incidents of `INC-0001`, a search with a new sentence. |
| `exercise_3_solution.py` | Exercise 3 — ask `llama3.2` about machine `M099` (does not exist): plain, then pushed until it invents, then grounded with the 40 lines of `machines.csv` and the rule "answer only from the context". |

LLM answers change from one machine to another. The scripts check keywords, a year pattern and counts, never an exact sentence.
