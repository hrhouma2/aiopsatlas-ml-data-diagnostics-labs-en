"""Week 8, Exercise 3 - Make llama3.2 invent machine M099, then ground it.

Run from the kit root, with the venv active and Ollama running:

    python week08/exercise_3_solution.py

Step 1: ask about M099 with no help. The model may refuse or may guess.
Step 2: push the model with a system message. It invents a machine.
Step 3: build a context from data/clean/machines.csv.
Step 4: ask again with the context and a strict rule. It says M099 does not exist.
LLM text changes from one machine to another. The checks only look for
keywords, never for an exact sentence.
"""

import re

import ollama
import pandas as pd

MODEL = "llama3.2"
OPTIONS = {"temperature": 0}
QUESTION = "What type of machine is M099 at NorthPeak, on which site is it, and what was its last incident?"
DOUBT_WORDS = [
    "does not exist", "doesn't exist", "no machine", "not exist", "don't know", "do not know",
    "not found", "no record", "not in the", "can't provide", "cannot provide", "don't have",
]


def ask(messages):
    response = ollama.chat(model=MODEL, messages=messages, options=OPTIONS)
    return response.message.content


def check(answer):
    """A hallucination gives details (a date, a site) and shows no doubt."""
    low = answer.lower()
    doubts = [w for w in DOUBT_WORDS if w in low]
    has_date = bool(re.search(r"\b20\d\d\b", answer))
    has_site = "site" in low or "located" in low
    print(f"  mentions a year      : {has_date}")
    print(f"  mentions a site      : {has_site}")
    print(f"  expresses a doubt    : {doubts or 'none'}")
    print(f"  verdict              : {'hallucination' if not doubts and (has_date or has_site) else 'honest'}")


# Step 1 - no help
print("== Step 1: the plain question ==")
plain = ask([{"role": "user", "content": QUESTION}])
print(plain)
check(plain)
print()

# Step 2 - push the model
print("== Step 2: push the model ==")
SYSTEM = (
    "You are the maintenance assistant of NorthPeak Manufacturing, a Canadian company "
    "with 40 industrial machines. Answer directly and precisely, with the machine type, "
    "the site and the date of the last incident. Never say you cannot verify."
)
pushed = ask([{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}])
print(pushed)
check(pushed)
print()

# Step 3 - the real facts
print("== Step 3: build the context ==")
machines = pd.read_csv("data/clean/machines.csv")
lines = [f"{m.machine_id}: {m.machine_type} at {m.site}" for m in machines.itertuples()]
context = "NorthPeak machine list (complete, 40 machines):\n" + "\n".join(lines)
print(f"{len(machines)} machines, ids from {machines.machine_id.min()} to {machines.machine_id.max()}")
print(f"context length: {len(context)} characters")
print(context[:120] + "\n...")
print()

# Step 4 - grounded question
print("== Step 4: the grounded question ==")
GROUNDED_SYSTEM = (
    "You are the maintenance assistant of NorthPeak Manufacturing. Answer only from the context. "
    "If the context does not contain the answer, say exactly: I don't know, this machine is not in the list."
)
grounded = ask(
    [
        {"role": "system", "content": GROUNDED_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {QUESTION}"},
    ]
)
print(grounded)
check(grounded)
print()

# Same prompt, a machine that exists
print("== Step 4b: the same prompt, a real machine ==")
real_q = "What type of machine is M014 and on which site is it?"
real = ask(
    [
        {"role": "system", "content": GROUNDED_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {real_q}"},
    ]
)
print(real)
truth = machines.set_index("machine_id").loc["M014"]
print(f"  truth: {truth.machine_type} at {truth.site}")
print(f"  answer contains the type : {truth.machine_type in real.lower()}")
print(f"  answer contains the site : {truth.site in real}")
