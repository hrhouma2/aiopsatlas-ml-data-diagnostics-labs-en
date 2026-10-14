"""Week 1, Exercise 2 - Talk to a local LLM.

Run from the kit root, with the venv active and Ollama running:

    python week01/exercise_2_solution.py

Step 1: a general question the model can answer from its training.
Step 2: a question about NorthPeak. The model has never seen our data.
Step 3: the same question, but we push the model to answer. It invents.
The LLM text changes from one machine to another. The checks only look
for keywords, never for an exact sentence.
"""

import ollama

MODEL = "llama3.2"
OPTIONS = {"temperature": 0}

# The truth, from data/clean/machines.csv (see Exercise 3).
REAL_COUNT = "40"
REAL_SITES = ["Montreal", "Quebec", "Toronto"]


def ask(messages):
    """Send one chat call and return the text of the answer."""
    response = ollama.chat(model=MODEL, messages=messages, options=OPTIONS)
    return response.message.content


def check(answer):
    """Say if the answer contains the real count and the real sites."""
    has_count = REAL_COUNT in answer
    sites_found = [s for s in REAL_SITES if s in answer]
    print(f"  contains the real count 40 : {has_count}")
    print(f"  real sites mentioned       : {sites_found or 'none'}")


# Step 1 - a general question
print("== Step 1: a general question ==")
general = ask([{"role": "user", "content": "What is a bearing in a machine? Answer in two sentences."}])
print(general)
print()

# Step 2 - a question about NorthPeak, no help
QUESTION = "How many machines does NorthPeak Manufacturing have and where are they?"
print("== Step 2: a question about NorthPeak ==")
plain = ask([{"role": "user", "content": QUESTION}])
print(plain)
check(plain)
print()

# Step 3 - the same question, but the model is told it works for NorthPeak
print("== Step 3: push the model to answer ==")
SYSTEM = (
    "You are the assistant of NorthPeak Manufacturing, a Canadian company "
    "that runs industrial machines. Answer questions about the company "
    "directly, with numbers."
)
pushed = ask([{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}])
print(pushed)
check(pushed)
print()
print("Real answer: 40 machines in Montreal, Quebec City and Toronto.")
