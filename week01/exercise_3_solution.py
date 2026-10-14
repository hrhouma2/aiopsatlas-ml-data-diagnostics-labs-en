"""Week 1, Exercise 3 - Prompt vs tool.

Run from the kit root, with the venv active and Ollama running:

    python week01/exercise_3_solution.py

A hand-made tool call, with no framework:
  1. prompt only: the model guesses a number (hallucination);
  2. a Python function reads the real count from data/clean/machines.csv;
  3. the model answers ONLY with JSON that names the tool and its argument;
  4. our code parses the JSON and runs the function;
  5. a second chat call gives the result back to the model;
  6. we compare the two answers.
"""

import json

import ollama
import pandas as pd

MODEL = "llama3.2"
OPTIONS = {"temperature": 0}
QUESTION = "How many machines does NorthPeak have in Toronto?"

machines = pd.read_csv("data/clean/machines.csv")


def count_machines(site):
    """Return the number of machines of one site, from the CSV."""
    return int((machines["site"] == site).sum())


# Step 1 - prompt only
print("== Step 1: prompt only ==")
PUSH = (
    "You are the assistant of NorthPeak Manufacturing, a Canadian company "
    "that runs industrial machines on three sites. Always answer with a "
    "precise number. Never say you cannot verify."
)
guess = ollama.chat(
    model=MODEL,
    messages=[{"role": "system", "content": PUSH}, {"role": "user", "content": QUESTION}],
    options=OPTIONS,
).message.content
print(guess)
print()

# Step 2 - the real numbers
print("== Step 2: the tool, in Python ==")
print(machines["site"].value_counts())
print("count_machines('Toronto') ->", count_machines("Toronto"))
print()

# Step 3 - ask the model for a tool call, as JSON only
print("== Step 3: the model asks for the tool ==")
TOOL_SYSTEM = """You can call one tool: count_machines(site).
Valid sites: Montreal, Quebec City, Toronto.
Answer ONLY with a JSON object like {"tool": "count_machines", "site": "Toronto"}.
No other text."""
first = ollama.chat(
    model=MODEL,
    messages=[{"role": "system", "content": TOOL_SYSTEM}, {"role": "user", "content": QUESTION}],
    format="json",
    options=OPTIONS,
)
raw = first.message.content
print("raw JSON from the model:", raw)

# Step 4 - our code runs the tool
print()
print("== Step 4: our code runs the tool ==")
call = json.loads(raw)
assert call["tool"] == "count_machines", call
result = count_machines(call["site"])
print(f"count_machines('{call['site']}') -> {result}")

# Step 5 - give the result back to the model
print()
print("== Step 5: the model reads the result ==")
second = ollama.chat(
    model=MODEL,
    messages=[
        {"role": "user", "content": QUESTION},
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": (
                f"Tool result: count_machines(site='{call['site']}') returned {result}. "
                "Answer the question in one sentence using only this number."
            ),
        },
    ],
    options=OPTIONS,
).message.content
print(second)

# Step 6 - compare
print()
print("== Step 6: compare ==")
print("prompt only :", guess.splitlines()[0])
print("with tool   :", second)
print("real count  :", count_machines("Toronto"))
print("tool answer contains the real count:", str(result) in second)
