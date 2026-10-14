"""Week 12, Exercise 2 - A diagnostic agent with two tools and a retry limit.

Run from the kit root, with the venv active and Ollama running:

    python week12/exercise_2_solution.py

The graph:
  agent -> (tool calls?) -> tools -> agent -> ... -> END
The state keeps the whole conversation (MessagesState) and a counter
`rounds`. After MAX_ROUNDS tool rounds the graph stops with an honest message.
"""

import sys

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

sys.path.insert(0, "week12")
from diag_tools import SCHEMA, TOOLS  # noqa: E402

MAX_ROUNDS = 3

SYSTEM = f"""You are the diagnostic assistant of NorthPeak Manufacturing.
You have two tools:
- query_database(sql): one read-only SQLite SELECT. {SCHEMA}
- search_manuals(question): the 6 maintenance manuals (thresholds, procedures, severity policy).
Stop limits, warning ranges and procedures are ONLY in the manuals.
Counts, dates, readings and the type of a machine are ONLY in the database. Never guess a machine's type.
Use a tool when you need facts. When you have the facts, answer in two or three short sentences.
Never invent numbers."""


# Step 1 - the state: messages + a counter
class AgentState(MessagesState):
    rounds: int


# Step 2 - the model with its tools
llm = ChatOllama(model="llama3.2", temperature=0).bind_tools(TOOLS)


def agent(state):
    reply = llm.invoke([SystemMessage(SYSTEM)] + state["messages"])
    # One tool per round: a small model that fires 3 calls at once guesses
    # the arguments of call 2 before it has seen the result of call 1.
    reply.tool_calls = reply.tool_calls[:1]
    return {"messages": [reply]}


# Step 3 - the tool node, wrapped to count the rounds
tool_node = ToolNode(TOOLS)


def run_tools(state):
    result = tool_node.invoke(state)
    return {"messages": result["messages"], "rounds": state["rounds"] + 1}


def give_up(state):
    text = "I stopped after the maximum number of tool rounds without a final answer. Please ask a simpler question."
    return {"messages": [AIMessage(content=text)]}


# Step 4 - the routing decision, then the graph
def build_graph(max_rounds=MAX_ROUNDS):
    def after_agent(state):
        last = state["messages"][-1]
        if not last.tool_calls:
            return END
        if state["rounds"] >= max_rounds:
            return "give_up"
        return "tools"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", run_tools)
    builder.add_node("give_up", give_up)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", after_agent, {"tools": "tools", "give_up": "give_up", END: END})
    builder.add_edge("tools", "agent")
    builder.add_edge("give_up", END)
    return builder.compile()


def ask(graph, question):
    """Run one question and print every step of the conversation."""
    print(f"\nQ: {question}")
    result = graph.invoke({"messages": [HumanMessage(question)], "rounds": 0})
    for message in result["messages"][1:]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print(f"  agent -> {call['name']}({call['args']})")
        elif message.type == "tool":
            first_line = message.content.splitlines()[0] if message.content else ""
            print(f"  tool  <- {first_line[:90]}")
        else:
            print(f"  answer: {message.content}")
    print(f"  rounds used: {result['rounds']}")
    return result


if __name__ == "__main__":
    graph = build_graph()
    print("== Step 4: the graph ==")
    print(graph.get_graph().draw_mermaid())

    print("\n== Step 5: three questions ==")
    ask(graph, "How many incidents does machine M014 have?")
    ask(graph, "At what vibration level must a compressor be stopped?")
    ask(graph, "Machine M014 vibrates at 9.5 mm/s. Is that above the stop limit for its type, and how many incidents does it already have?")

    print("\n== Step 6: the same questions, written for a small model ==")
    ask(graph, "According to the manuals, at what vibration level must a compressor be stopped?")
    ask(graph, "Step 1: look up the machine_type of M014 in the database. Step 2: search the manuals for the vibration stop limit of that type. Then tell me if 9.5 mm/s is above it.")

    print("\n== Step 7: prove the limit with max_rounds = 0 ==")
    strict = build_graph(max_rounds=0)
    ask(strict, "How many incidents does machine M014 have?")
