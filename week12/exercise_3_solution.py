"""Week 12, Exercise 3 - Memory on disk and a human approval step.

Run from the kit root, with the venv active and Ollama running:

    python week12/exercise_3_solution.py

Two additions to the agent of Exercise 2:
  1. a SqliteSaver checkpointer: the whole state is saved after every node in
     data/checkpoints.sqlite, under a thread_id;
  2. an approval gate: before any SQL that touches the `incidents` table the
     graph pauses with interrupt(). A human answers, the graph resumes with
     Command(resume=True) or Command(resume=False).
"""

import sqlite3
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

sys.path.insert(0, "week12")
from exercise_2_solution import MAX_ROUNDS, AgentState, agent, give_up, run_tools  # noqa: E402

CHECKPOINT_DB = Path("data/checkpoints.sqlite")


# Step 1 - the approval gate
def needs_approval(call):
    return call["name"] == "query_database" and "incidents" in call["args"].get("sql", "").lower()


def approval_gate(state):
    last = state["messages"][-1]
    risky = [call for call in last.tool_calls if needs_approval(call)]
    if not risky:
        return {}
    approved = interrupt(
        {"question": "This query reads the incidents table. Run it?", "sql": risky[0]["args"]["sql"]}
    )
    if approved:
        return {}
    refusals = [
        ToolMessage(content="The human refused this query. Answer without it.", tool_call_id=call["id"])
        for call in last.tool_calls
    ]
    return {"messages": refusals}


# Step 2 - the routing around the gate
def after_agent(state):
    last = state["messages"][-1]
    if not last.tool_calls:
        return END
    if state["rounds"] >= MAX_ROUNDS:
        return "give_up"
    return "gate"


def after_gate(state):
    refused = isinstance(state["messages"][-1], ToolMessage)
    return "agent" if refused else "tools"


# Step 3 - the graph with a checkpointer
def build_graph(checkpointer):
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("gate", approval_gate)
    builder.add_node("tools", run_tools)
    builder.add_node("give_up", give_up)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", after_agent, {"gate": "gate", "give_up": "give_up", END: END})
    builder.add_conditional_edges("gate", after_gate, {"agent": "agent", "tools": "tools"})
    builder.add_edge("tools", "agent")
    builder.add_edge("give_up", END)
    return builder.compile(checkpointer=checkpointer)


def show(result):
    """Print the pause request, or the final answer."""
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"  PAUSED  : {payload['question']}")
        print(f"  SQL     : {payload['sql']}")
    else:
        print(f"  answer  : {result['messages'][-1].content}")


def print_history(graph, thread):
    """One line per message saved in the thread."""
    for message in graph.get_state(thread).values["messages"]:
        if message.type == "ai" and message.tool_calls:
            print(f"    ai    -> {message.tool_calls[0]['name']}({message.tool_calls[0]['args']})")
        else:
            print(f"    {message.type:5} :  {message.content[:70].replace(chr(10), ' / ')}")


if __name__ == "__main__":
    CHECKPOINT_DB.unlink(missing_ok=True)  # start clean so the run below is repeatable
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    graph = build_graph(SqliteSaver(conn))
    thread = {"configurable": {"thread_id": "technician-1"}}

    # Step 4 - run 1: the gate pauses, the human approves
    print("== Run 1, thread technician-1 ==")
    print("Q: How many incidents does machine M014 have?")
    result = graph.invoke({"messages": [HumanMessage("How many incidents does machine M014 have?")], "rounds": 0}, thread)
    show(result)
    print("  next node:", graph.get_state(thread).next)
    result = graph.invoke(Command(resume=True), thread)
    show(result)

    # Step 5 - run 2, same thread: the model remembers M014
    print("\n== Run 2, same thread: a follow-up question ==")
    print("Q: And what is its machine type?")
    result = graph.invoke({"messages": [HumanMessage("And what is its machine type?")], "rounds": 0}, thread)
    show(result)
    if "__interrupt__" in result:
        result = graph.invoke(Command(resume=True), thread)
        show(result)
    print("  messages saved in the thread:", len(graph.get_state(thread).values["messages"]))
    print_history(graph, thread)

    # Step 6 - a new thread has no memory
    print("\n== Run 3, new thread technician-2: same follow-up ==")
    other = {"configurable": {"thread_id": "technician-2"}}
    print("Q: And what is its machine type?")
    result = graph.invoke({"messages": [HumanMessage("And what is its machine type?")], "rounds": 0}, other)
    show(result)
    if "__interrupt__" in result:
        result = graph.invoke(Command(resume=True), other)
        show(result)

    # Step 7 - the human refuses
    print("\n== Run 4, thread technician-3: the human refuses ==")
    third = {"configurable": {"thread_id": "technician-3"}}
    print("Q: List the incidents of machine M001.")
    result = graph.invoke({"messages": [HumanMessage("List the incidents of machine M001.")], "rounds": 0}, third)
    show(result)
    if "__interrupt__" in result:
        result = graph.invoke(Command(resume=False), third)
        show(result)

    rows = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
    print(f"\ncheckpoints saved in {CHECKPOINT_DB}: {rows}")
    conn.close()
