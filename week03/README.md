# Week 3 — Data sources and database types: solution scripts

Run every script from the kit root, with the venv active and `python data/make_dataset.py` done. No LLM is needed this week.

| Script | What it does |
| --- | --- |
| `exercise_1_solution.py` | Exercise 1 — Three tables, two tools, same numbers: `incidents JOIN machines JOIN readings` on `data/diagnostics.sqlite`, then the same with pandas `merge`; incidents and cost by site (66 / 56 / 33), by site and type, mean temperature on the incident day; `assert` that SQL and pandas agree. |
| `exercise_2_solution.py` | Exercise 2 — JSON documents, two ways: `json.load` on `data/json/incidents.json`, nested access `doc["machine"]["site"]`, tag filter (`high` + `pump` gives 4 ids), then DuckDB `read_json_auto` with `machine.site` and `list_contains`; ends with a five-line dict cache. |
| `exercise_3_solution.py` | Exercise 3 — Walk the graph with merges: `data/graph/nodes.csv` (207) and `edges.csv` (390); one table per relation, 2-hop and 3-hop paths with `merge`; bearing incidents by site (Toronto 22, Montreal 18, Quebec City 8); most connected machine `M017`, degree 10. |

Every number printed by these scripts is the number quoted in the lessons. If yours differs, run `python data/make_dataset.py` again.
