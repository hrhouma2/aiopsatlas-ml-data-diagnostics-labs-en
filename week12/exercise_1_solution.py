"""Week 12, Exercise 1 - A first graph: the router.

Run from the kit root, with the venv active and Ollama running:

    python week12/exercise_1_solution.py

One state, three nodes, one conditional edge:
  classify  -> "data" or "manual"
  sql_node  -> writes one SELECT, runs it read-only, answers from the rows
  rag_node  -> searches the manuals, answers from the 3 best passages
"""

import sys
from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, "week12")
from diag_tools import SCHEMA, retrieve, run_sql  # noqa: E402

llm = ChatOllama(model="llama3.2", temperature=0)


# Step 1 - the state: one dictionary that every node reads and updates
class RouterState(TypedDict):
    question: str
    route: str
    sql: str
    context: str
    answer: str


# Step 2 - the classifier node
CLASSIFY_PROMPT = """Classify the question. Answer with exactly one word.
- data   : the answer needs numbers, counts, dates or rows from the database
- manual : the answer is a procedure, a rule, a threshold or a definition from the manuals
Question: {question}
One word (data or manual):"""


def classify(state):
    reply = llm.invoke(CLASSIFY_PROMPT.format(question=state["question"]))
    word = reply.content.strip().lower().strip(".")
    route = "manual" if "manual" in word else "data"
    return {"route": route}


# Step 3 - the two worker nodes
SQL_PROMPT = """{schema}

Write ONE SQLite SELECT that answers the question. Reply with the SQL only, no explanation, no markdown.
Only the machines table has site and machine_type: JOIN machines ON machine_id when you need them.
Question: {question}"""

ANSWER_PROMPT = """Answer the question in one or two short sentences, using ONLY the information below.
If the information is not enough, say "I don't know".

Question: {question}

Information:
{context}"""


def sql_node(state):
    sql = llm.invoke(SQL_PROMPT.format(schema=SCHEMA, question=state["question"])).content
    sql = sql.strip().strip("`").removeprefix("sql").strip().rstrip(";")
    rows = run_sql(sql)
    context = (
        f"The database was queried with:\n{sql}\n"
        f"It returned (first line = column names):\n{rows}\n"
        "These rows already answer the question. Read the numbers from them."
    )
    answer = llm.invoke(ANSWER_PROMPT.format(question=state["question"], context=context)).content
    return {"sql": sql, "context": rows, "answer": answer}


def rag_node(state):
    docs = retrieve(state["question"], k=3)
    context = "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)
    answer = llm.invoke(ANSWER_PROMPT.format(question=state["question"], context=context)).content
    sources = ", ".join(sorted({d.metadata["source"] for d in docs}))
    return {"sql": "", "context": sources, "answer": answer}


# Step 4 - wire the graph
def choose_route(state):
    return state["route"]


builder = StateGraph(RouterState)
builder.add_node("classify", classify)
builder.add_node("sql_node", sql_node)
builder.add_node("rag_node", rag_node)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", choose_route, {"data": "sql_node", "manual": "rag_node"})
builder.add_edge("sql_node", END)
builder.add_edge("rag_node", END)
graph = builder.compile()

if __name__ == "__main__":
    print("== Step 4: the graph as Mermaid ==")
    print(graph.get_graph().draw_mermaid())

    # Step 5 - run four questions through the router
    QUESTIONS = [
        "How many incidents happened in Toronto?",
        "When should a pump be stopped for vibration?",
        "Which machine has the most incidents?",
        "What severity does a sensor fault get?",
    ]
    print("== Step 5: four questions ==")
    for question in QUESTIONS:
        result = graph.invoke({"question": question})
        print(f"\nQ: {question}")
        print(f"route  : {result['route']}")
        if result["route"] == "data":
            print(f"sql    : {result['sql']}")
            print(f"rows   : {result['context'].splitlines()[:3]}")
        else:
            print(f"sources: {result['context']}")
        print(f"answer : {result['answer']}")
