"""Week 9, Exercise 2 - Keyword search vs vector search vs a simple hybrid.

Run from the kit root, with the venv active, Ollama running and the Chroma
store of Exercise 1 built (the script rebuilds it if data/chroma is empty):

    python week09/exercise_2_solution.py

Step 1: load the 31 chunks and the store.
Step 2: keyword search with TF-IDF for one question.
Step 3: vector search for the same question.
Step 4: five questions, side by side.
Step 5: a hybrid score with reciprocal rank fusion.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so that `common` is importable
from common.rag import build_or_load_manuals_store, chunk_manuals  # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 44)

QUESTIONS = [
    "At what vibration level must I stop a pump?",
    "Who is the site supervisor in Toronto?",
    "What do fault codes E40 to E59 mean?",
    "The machine gets hot when it is warm outside. Is that normal?",
    "How long should I wait before I restart a pump after a stop?",
]

# Step 1 - the chunks and the store
print("== Step 1: chunks and store ==")
texts, metadatas, ids = chunk_manuals()
store = build_or_load_manuals_store()
print(f"{len(texts)} chunks, {store._collection.count()} vectors in data/chroma")
print()


def short_id(chunk_id):
    """'01-pump-troubleshooting.md-2' -> '01-pump-2'"""
    name, i = chunk_id.rsplit("-", 1)
    return "-".join(name.split("-")[:2]) + "-" + i


# Step 2 - keyword search with TF-IDF
print("== Step 2: keyword search ==")
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
chunk_matrix = vectorizer.fit_transform(texts)
print(f"TF-IDF matrix: {chunk_matrix.shape}")


def keyword_scores(question):
    q = vectorizer.transform([question])
    return (chunk_matrix @ q.T).toarray().ravel()


scores = keyword_scores(QUESTIONS[0])
order = np.argsort(scores)[::-1]
print(f"question: {QUESTIONS[0]}")
for i in order[:3]:
    print(f"  {scores[i]:.3f}  {short_id(ids[i])}  {texts[i].splitlines()[0][:60]!r}")
print(f"chunks with a score of exactly 0: {(scores == 0).sum()} of {len(texts)}")
print()

# Step 3 - vector search
print("== Step 3: vector search ==")
hits = store.similarity_search_with_score(QUESTIONS[0], k=3)
for doc, dist in hits:
    print(f"  {dist:.3f}  {doc.metadata['source']}  {doc.page_content.splitlines()[0][:60]!r}")
print("(Chroma distance: smaller is closer)")
print()

# Build a lookup from chunk text to chunk id, to compare both searches.
id_of_text = {t: i for t, i in zip(texts, ids)}


def vector_ranked_ids(question):
    hits = store.similarity_search_with_score(question, k=len(texts))
    return [id_of_text[doc.page_content] for doc, _ in hits]


def keyword_ranked_ids(question):
    scores = keyword_scores(question)
    return [ids[i] for i in np.argsort(scores)[::-1]]


# Step 4 - five questions side by side
print("== Step 4: five questions ==")
rows = []
for q in QUESTIONS:
    kw = keyword_ranked_ids(q)
    vec = vector_ranked_ids(q)
    rows.append({"question": q, "keyword top-1": short_id(kw[0]), "vector top-1": short_id(vec[0]),
                 "kw score": round(float(keyword_scores(q).max()), 3)})
print(pd.DataFrame(rows).to_string(index=False))
print()


# Step 5 - hybrid with reciprocal rank fusion
def rrf(rankings, k=1):
    """Reciprocal rank fusion: each list gives 1 / (k + rank) to every id."""
    score = {}
    for ranked in rankings:
        for rank, chunk_id in enumerate(ranked):
            score[chunk_id] = score.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(score.items(), key=lambda t: -t[1])


print("== Step 5: hybrid ranking ==")
q = QUESTIONS[3]
kw = keyword_ranked_ids(q)
vec = vector_ranked_ids(q)
print(f"question: {q}")
print(f"  keyword top-3: {[short_id(i) for i in kw[:3]]}")
print(f"  vector  top-3: {[short_id(i) for i in vec[:3]]}")
fused = rrf([kw, vec])
print("  hybrid  top-3:", [(short_id(i), round(s, 3)) for i, s in fused[:3]])
print()
print("hybrid top-1 for the five questions:")
for q in QUESTIONS:
    fused = rrf([keyword_ranked_ids(q), vector_ranked_ids(q)])
    print(f"  {short_id(fused[0][0]):<14} {q}")
