"""Week 13, Exercise 2 - Evaluate the RAG on a golden set of 10 questions.

Run from the kit root, with the venv active and Ollama running:

    python week13/exercise_2_solution.py

week13/golden.json holds 10 questions the 6 manuals can answer, each with the
manual that holds the answer and one or more keywords the answer must contain.
Two scores:
  hit rate      - the expected source is among the 3 retrieved chunks
  keyword rate  - at least one keyword appears in the model's answer
"""

import json
import sys
from pathlib import Path

import pandas as pd
from langchain_ollama import ChatOllama

sys.path.insert(0, "week13")
from diag_tools import retrieve  # noqa: E402

GOLDEN_PATH = Path("week13/golden.json")
RESULTS_PATH = Path("week13/eval_results.csv")
llm = ChatOllama(model="llama3.2", temperature=0)

ANSWER_PROMPT = """Answer the question in one short sentence, using ONLY the passages below.
If the passages do not contain the answer, say "I don't know".

Question: {question}

Passages:
{context}"""


# Step 2 - retrieval only: is the right manual in the top 3?
def retrieval_hit(question, expected_source, k=3):
    docs = retrieve(question, k=k)
    sources = [d.metadata["source"] for d in docs]
    return expected_source in sources, docs


# Step 3 - the answer: does it contain a keyword?
def answer_from(question, docs):
    context = "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)
    return llm.invoke(ANSWER_PROMPT.format(question=question, context=context)).content.strip()


def keyword_hit(answer, keywords):
    return any(keyword.lower() in answer.lower() for keyword in keywords)


# Step 4 - the loop over the golden set
def evaluate(golden):
    rows = []
    for item in golden:
        hit, docs = retrieval_hit(item["question"], item["expected_source"])
        answer = answer_from(item["question"], docs)
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_source": item["expected_source"],
                "rank": next((i + 1 for i, d in enumerate(docs) if d.metadata["source"] == item["expected_source"]), None),
                "source_hit": hit,
                "keyword_hit": keyword_hit(answer, item["keywords"]),
                "answer": answer,
            }
        )
        print(f"{item['id']:>2}. source={'HIT ' if hit else 'MISS'} keyword={'HIT ' if rows[-1]['keyword_hit'] else 'MISS'}  {answer[:80]}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    print(f"== Step 1: {len(golden)} golden questions ==")
    print(golden[0])

    print("\n== Steps 2-4: evaluate ==")
    results = evaluate(golden)
    results.to_csv(RESULTS_PATH, index=False)

    print("\n== Step 5: the scores ==")
    print(f"hit rate (expected source in top 3): {results['source_hit'].mean():.0%}")
    print(f"keyword rate (keyword in the answer): {results['keyword_hit'].mean():.0%}")
    print("\nrank of the expected source (None = not in top 3):")
    print(results[["id", "expected_source", "rank"]].to_string(index=False))
    misses = results[~results["source_hit"] | ~results["keyword_hit"]]
    if len(misses):
        print("\nquestions to look at:")
        for row in misses.itertuples():
            print(f"  {row.id}. {row.question}\n     -> {row.answer}")
    print(f"\nsaved {RESULTS_PATH}")
