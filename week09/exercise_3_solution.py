"""Week 9, Exercise 3 - A first RAG answer with citations.

Run from the kit root, with the venv active, Ollama running (`llama3.2` and
`nomic-embed-text` pulled) and the Chroma store of Exercise 1 built (the
script rebuilds it if data/chroma is empty):

    python week09/exercise_3_solution.py

Step 1: retrieve the 3 closest chunks for a question.
Step 2: build the context with a [source: file] tag on each chunk.
Step 3: the grounded prompt.
Step 4: ask llama3.2 and check the citation.
Step 5: a question the manuals cannot answer.
LLM text changes from one machine to another. The checks only look for
keywords, never for an exact sentence.
"""

import sys
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so that `common` is importable
from common.rag import build_or_load_manuals_store  # noqa: E402

MODEL = "llama3.2"
OPTIONS = {"temperature": 0}
RULES = (
    "Answer only from the context. If the context does not contain the answer, "
    "say: I don't know. Cite the source file in brackets."
)

store = build_or_load_manuals_store()

# Step 1 - retrieve
print("== Step 1: retrieve ==")
question = "A pump vibrates at 7.5 mm/s for three days. What should I do?"
docs = store.similarity_search(question, k=3)
for d in docs:
    print(f"  {d.metadata['source']}  {d.page_content.splitlines()[0][:60]!r}")
print()


# Step 2 - context with source tags
def build_context(docs):
    parts = [f"[source: {d.metadata['source']}]\n{d.page_content}" for d in docs]
    return "\n\n---\n\n".join(parts)


context = build_context(docs)
print("== Step 2: the context ==")
print(f"{len(context)} characters, {len(docs)} chunks")
print(context[:160] + "\n...")
print()


# Step 3 and 4 - the grounded prompt
def rag_answer(question, k=3):
    docs = store.similarity_search(question, k=k)
    prompt = f"{RULES}\n\nContext:\n{build_context(docs)}\n\nQuestion: {question}"
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}], options=OPTIONS)
    return response.message.content, [d.metadata["source"] for d in docs]


def check(answer, sources, must_contain=(), expect_unknown=False):
    low = answer.lower()
    cited = [s for s in set(sources) if s in answer]
    print(f"  cites a retrieved file : {cited or 'none'}")
    if expect_unknown:
        unknown = "don't know" in low or "do not know" in low
        print(f"  says I don't know      : {unknown}")
    for word in must_contain:
        print(f"  contains '{word}'{' ' * (13 - len(word))}: {word in low}")


print("== Step 3-4: the grounded answer ==")
answer, sources = rag_answer(question)
print(answer)
check(answer, sources, must_contain=["stop", "high"])
print()

print("== Another question, policy manual ==")
q2 = "Who must be notified for a high severity incident?"
answer, sources = rag_answer(q2)
print(answer)
check(answer, sources, must_contain=["supervisor", "safety"])
print()

# Step 5 - a question the manuals cannot answer
print("== Step 5: a question with no answer in the manuals ==")
q3 = "How much does a replacement bearing for a pump cost in Canadian dollars?"
answer, sources = rag_answer(q3)
print(f"retrieved: {sources}")
print(answer)
check(answer, sources, expect_unknown=True)
