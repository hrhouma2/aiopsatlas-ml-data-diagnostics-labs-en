# Week 12 — Stateful workflows with LangGraph: solution scripts

Run every script from the kit root, with the venv active and Ollama running (`llama3.2` and `nomic-embed-text` pulled).

| Script | What it does |
| --- | --- |
| `diag_tools.py` | The two tools shared by the exercises: `query_database(sql)` (read-only SQLite, one SELECT, `LIMIT 50`) and `search_manuals(question)` (top-3 chunks of the 6 manuals from the Chroma store `data/chroma`). Weeks 9 to 11 built the same pieces; this file only puts them together. Run it alone to check both tools. |
| `exercise_1_solution.py` | Exercise 1 — A first graph: a classifier node answers `data` or `manual`, a conditional edge sends the question to a SQL node or a RAG node, then END. Prints the graph as Mermaid and runs four questions. |
| `exercise_2_solution.py` | Exercise 2 — Diagnostic agent: `bind_tools` + `ToolNode`, the conversation in `MessagesState`, a `rounds` counter with a hard limit of 3 tool rounds and a `give_up` node. One tool per round. |
| `exercise_3_solution.py` | Exercise 3 — Memory and approval: `SqliteSaver` checkpointer in `data/checkpoints.sqlite` with a `thread_id`, and an `interrupt()` gate before any SQL on the `incidents` table, resumed with `Command(resume=True)` or `Command(resume=False)`. Two runs on the same thread prove the memory. |

Files created by the scripts: `data/checkpoints.sqlite` (deleted and rebuilt at each run of Exercise 3). The Chroma store `data/chroma` is created once and reused.

LLM answers change from one machine to another. `llama3.2` is a 3B model: it sometimes picks the wrong tool or guesses a machine type. The exercises show these real outputs and how to write questions that a small model can follow.
