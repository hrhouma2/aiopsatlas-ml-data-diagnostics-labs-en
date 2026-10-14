"""Week 8, Exercise 2 - Embed the 155 incident descriptions and find the nearest ones.

Run from the kit root, with the venv active, Ollama running and
`ollama pull nomic-embed-text` done:

    python week08/exercise_2_solution.py

Step 1: embed one sentence and look at the vector.
Step 2: embed all 155 descriptions in one call.
Step 3: cosine similarity with numpy.
Step 4: the 3 nearest incidents of INC-0001.
Step 5: search the incidents with a sentence that is not in the file.
"""

import numpy as np
import ollama
import pandas as pd

pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", 90)

EMBED_MODEL = "nomic-embed-text"

incidents = pd.read_csv("data/clean/incidents.csv")
texts = incidents["description"].tolist()

# Step 1 - one vector
print("== Step 1: one vector ==")
one = ollama.embed(model=EMBED_MODEL, input=[texts[0]])
vector = np.array(one.embeddings[0])
print(texts[0])
print(f"tokens read   : {one.prompt_eval_count}")
print(f"vector length : {len(vector)}")
print(f"first 5 values: {np.round(vector[:5], 4).tolist()}")
print(f"norm          : {np.linalg.norm(vector):.4f}")
print()

# Step 2 - all 155 vectors
print("== Step 2: all 155 descriptions ==")
response = ollama.embed(model=EMBED_MODEL, input=texts)
E = np.array(response.embeddings)
print(f"matrix shape: {E.shape}")
print()

# Step 3 - cosine similarity
print("== Step 3: cosine similarity ==")
E_norm = E / np.linalg.norm(E, axis=1, keepdims=True)
S = E_norm @ E_norm.T
print(f"similarity matrix shape: {S.shape}")
print(f"INC-0001 with itself   : {S[0, 0]:.3f}")
print(f"INC-0001 with INC-0002 : {S[0, 1]:.3f}")
print(f"INC-0001 with INC-0005 : {S[0, 4]:.3f}")
print()

# Step 4 - the 3 nearest incidents of INC-0001
print("== Step 4: 3 nearest of INC-0001 ==")
print(f"query: {incidents.loc[0, 'incident_id']} [{incidents.loc[0, 'category']}] {texts[0]}")
order = np.argsort(S[0])[::-1]
nearest = [i for i in order if i != 0][:3]
for i in nearest:
    row = incidents.loc[i]
    print(f"  {S[0, i]:.3f}  {row.incident_id} [{row.category}] {row.description}")
print()

# Step 5 - search with a new sentence
print("== Step 5: search with a new sentence ==")
query = "the fan is running but dust is blocking the air flow"
q = np.array(ollama.embed(model=EMBED_MODEL, input=[query]).embeddings[0])
q = q / np.linalg.norm(q)
scores = E_norm @ q
top = np.argsort(scores)[::-1][:3]
print(f"query: {query}")
for i in top:
    row = incidents.loc[i]
    print(f"  {scores[i]:.3f}  {row.incident_id} [{row.category}] {row.description}")
print()
print("categories of the 3 hits:", incidents.loc[top, "category"].tolist())
