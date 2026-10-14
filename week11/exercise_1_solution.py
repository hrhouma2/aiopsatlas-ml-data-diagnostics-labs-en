"""Week 11, Exercise 1 - Tool calling with llama3.2.

Run from the kit root, with the venv active and Ollama running:

    python week11/exercise_1_solution.py

Steps:
  1. a Python function that reads data/diagnostics.sqlite (read-only);
  2. give the function to the model as a tool: the model asks, it does not run;
  3. check the arguments, then run the tool in our code;
  4. send the result back as a "tool" message, get the final sentence;
  5. the trap: a question that needs no tool still triggers a tool call;
  6. the same loop with LangChain's bind_tools().
"""

import json
import re
import sqlite3

import ollama
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

MODEL = "llama3.2"
OPTIONS = {"temperature": 0}
DB_URI = "file:data/diagnostics.sqlite?mode=ro"
MACHINE_ID = re.compile(r"^M\d{3}$")


# Step 1 - the tool is a normal function
def get_machine_stats(machine_id: str) -> dict:
    """Return the number of readings, the mean temperature, the max vibration
    and the number of incidents of one machine, for example 'M014'."""
    con = sqlite3.connect(DB_URI, uri=True)
    n_readings, mean_temp, max_vib = con.execute(
        "SELECT COUNT(*), AVG(temperature_c), MAX(vibration_mm_s) FROM readings WHERE machine_id = ?",
        (machine_id,),
    ).fetchone()
    n_incidents = con.execute(
        "SELECT COUNT(*) FROM incidents WHERE machine_id = ?", (machine_id,)
    ).fetchone()[0]
    con.close()
    return {
        "machine_id": machine_id,
        "readings": n_readings,
        "mean_temperature_c": round(mean_temp, 1) if mean_temp is not None else None,
        "max_vibration_mm_s": max_vib,
        "incidents": n_incidents,
    }


print("== Step 1: the tool, in Python ==")
print(get_machine_stats("M014"))
print()

# Step 2 - the model asks for the tool
print("== Step 2: the model asks ==")
QUESTION = "Give me the stats of machine M014."
messages = [{"role": "user", "content": QUESTION}]
first = ollama.chat(model=MODEL, messages=messages, tools=[get_machine_stats], options=OPTIONS)
print("content   :", repr(first.message.content))
print("tool_calls:", first.message.tool_calls)
print()

# Step 3 - our code checks and runs
print("== Step 3: our code runs the tool ==")
TOOLS = {"get_machine_stats": get_machine_stats}


def run_tool_call(call):
    """Run one tool call only if the name and the argument are valid."""
    name = call.function.name
    args = call.function.arguments
    if name not in TOOLS:
        return {"error": f"unknown tool {name}"}
    machine_id = str(args.get("machine_id", ""))
    if not MACHINE_ID.match(machine_id):
        return {"error": f"invalid machine_id {machine_id!r}, expected M001 to M040"}
    return TOOLS[name](machine_id)


result = run_tool_call(first.message.tool_calls[0])
print(result)
print()

# Step 4 - the model reads the result
print("== Step 4: the model reads the result ==")
messages.append(first.message)
messages.append(
    {"role": "tool", "tool_name": "get_machine_stats", "content": json.dumps(result)}
)
messages.append(
    {"role": "user", "content": "Answer the question in one sentence, using only these numbers."}
)
final = ollama.chat(model=MODEL, messages=messages, options=OPTIONS)
print(final.message.content)
print("contains the real incident count:", str(result["incidents"]) in final.message.content)
print()

# Step 5 - the trap: a tool call that should not happen
print("== Step 5: the trap ==")
greeting = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "Hello, who are you?"}],
    tools=[get_machine_stats],
    options=OPTIONS,
)
print("tool_calls:", greeting.message.tool_calls)
if greeting.message.tool_calls:
    print("our check :", run_tool_call(greeting.message.tool_calls[0]))
print()

# Step 6 - the same with LangChain
print("== Step 6: with LangChain bind_tools ==")
lc_tool = tool(get_machine_stats)
llm = ChatOllama(model=MODEL, temperature=0).bind_tools([lc_tool])
ai = llm.invoke(QUESTION)
print("tool_calls:", ai.tool_calls)
lc_result = lc_tool.invoke(ai.tool_calls[0]["args"])
print("tool result:", lc_result)
