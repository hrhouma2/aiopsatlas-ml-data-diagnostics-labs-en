"""Week 13, Exercise 1 - Trace every step of the agent to a JSONL file.

Run from the kit root, with the venv active and Ollama running:

    python week13/exercise_1_solution.py

The agent is the one of Week 12 (two tools, a round counter). The new part is
`graph.stream(..., stream_mode="updates")`: every node gives us its update as
soon as it finishes. We turn each update into one JSON line in
week13/trace.jsonl, then read the file back with pandas.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

sys.path.insert(0, "week13")
from diag_tools import SCHEMA, TOOLS  # noqa: E402

TRACE_PATH = Path("week13/trace.jsonl")
MAX_ROUNDS = 3
SYSTEM = f"""You are the diagnostic assistant of NorthPeak Manufacturing.
You have two tools:
- query_database(sql): one read-only SQLite SELECT. {SCHEMA}
- search_manuals(question): the 6 maintenance manuals (thresholds, procedures, severity policy).
Stop limits, warning ranges and procedures are ONLY in the manuals.
Counts, dates, readings and the type of a machine are ONLY in the database. Never guess a machine's type.
Use a tool when you need facts. When you have the facts, answer in two or three short sentences.
Never invent numbers."""


# ------------------------------------------------ the Week 12 agent, unchanged
class AgentState(MessagesState):
    rounds: int


llm = ChatOllama(model="llama3.2", temperature=0).bind_tools(TOOLS)
tool_node = ToolNode(TOOLS)


def agent(state):
    reply = llm.invoke([SystemMessage(SYSTEM)] + state["messages"])
    reply.tool_calls = reply.tool_calls[:1]  # one tool per round
    return {"messages": [reply]}


def run_tools(state):
    result = tool_node.invoke(state)
    return {"messages": result["messages"], "rounds": state["rounds"] + 1}


def give_up(state):
    return {"messages": [AIMessage(content="I stopped after the maximum number of tool rounds.")]}


def after_agent(state):
    last = state["messages"][-1]
    if not last.tool_calls:
        return END
    return "give_up" if state["rounds"] >= MAX_ROUNDS else "tools"


builder = StateGraph(AgentState)
builder.add_node("agent", agent)
builder.add_node("tools", run_tools)
builder.add_node("give_up", give_up)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", after_agent, {"tools": "tools", "give_up": "give_up", END: END})
builder.add_edge("tools", "agent")
builder.add_edge("give_up", END)
graph = builder.compile()


# ---------------------------------------------------------- Step 2: one record
def make_record(question, node, payload, duration_ms):
    """Turn one node update into a flat dictionary (one line of the trace)."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "question": question,
        "node": node,
        "duration_ms": round(duration_ms),
        "tool": None,
        "prompt_tokens": None,
        "output_tokens": None,
        "tokens_per_s": None,
    }
    for message in payload.get("messages", []):
        if message.type == "ai":
            meta = message.response_metadata
            record["prompt_tokens"] = meta.get("prompt_eval_count")
            record["output_tokens"] = meta.get("eval_count")
            if meta.get("eval_duration"):
                record["tokens_per_s"] = round(meta["eval_count"] / (meta["eval_duration"] / 1e9), 1)
            if message.tool_calls:
                record["tool"] = message.tool_calls[0]["name"]
                record["args"] = message.tool_calls[0]["args"]
        elif message.type == "tool":
            record["tool"] = message.name
            record["result_chars"] = len(message.content)
    return record


# --------------------------------------------- Step 3: stream and write lines
def traced_run(question):
    with TRACE_PATH.open("a", encoding="utf-8") as log:
        last = time.perf_counter()
        for update in graph.stream({"messages": [HumanMessage(question)], "rounds": 0}, stream_mode="updates"):
            now = time.perf_counter()
            for node, payload in update.items():
                record = make_record(question, node, payload, (now - last) * 1000)
                log.write(json.dumps(record) + "\n")
                print(f"  {record['node']:7} {record['duration_ms']:>6} ms  tool={record['tool']}")
                if node == "agent" and record["tool"] is None:
                    print(f"  answer: {payload['messages'][-1].content}")
            last = now


if __name__ == "__main__":
    TRACE_PATH.unlink(missing_ok=True)
    QUESTIONS = [
        "How many incidents does machine M014 have?",
        "According to the manuals, at what vibration level must a compressor be stopped?",
        "Step 1: look up the machine_type of M014 in the database. Step 2: search the manuals for the vibration stop limit of that type. Then tell me if 9.5 mm/s is above it.",
    ]
    print("== Step 3: stream three questions ==")
    for question in QUESTIONS:
        print(f"Q: {question[:70]}")
        traced_run(question)

    # Step 4 - read one trace line
    print("\n== Step 4: the first line of week13/trace.jsonl ==")
    print(TRACE_PATH.read_text(encoding="utf-8").splitlines()[0])

    # Step 5 - read the whole trace with pandas
    print("\n== Step 5: the trace as a table ==")
    trace = pd.read_json(TRACE_PATH, lines=True)
    print(trace[["node", "tool", "duration_ms", "prompt_tokens", "output_tokens", "tokens_per_s"]].to_string())
    print("\nlines per node:")
    print(trace["node"].value_counts().to_string())
    print("\ntotal ms per question:")
    print(trace.groupby(trace["question"].str[:45], sort=False)["duration_ms"].sum().to_string())
    print(f"\nmean tokens/s of the model: {trace['tokens_per_s'].mean():.1f}")
