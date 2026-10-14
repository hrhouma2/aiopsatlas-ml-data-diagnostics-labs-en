# Week 2 — Python data retrieval and exploration: solution scripts

Run every script from the kit root, with the venv active and `python data/make_dataset.py` done. No LLM is needed this week.

| Script | What it does |
| --- | --- |
| `exercise_1_solution.py` | Exercise 1 — Explore the raw readings: `info`, `isna`, `duplicated`, `describe` on `data/raw/readings_raw.csv`; prints the defect table (14,750 rows, 150 duplicates, 291 missing temperatures after the drop, 15 × 999.0, 12 negative loads, 10 vibration spikes). |
| `exercise_2_solution.py` | Exercise 2 — Six SQL queries, twice: `COUNT`, `WHERE`, `GROUP BY`, `AVG`, `JOIN`, `SUM` on `data/diagnostics.sqlite`, and the same six in pandas on `data/clean/*.csv`, with `assert` that both agree. |
| `exercise_3_solution.py` | Exercise 3 — Clean the raw readings: drop duplicates, fix the sign of the loads, divide the vibration spikes by 25, turn 999.0 into missing, fill with the machine median; every step checked against `data/clean/readings.csv`; writes `data/clean/readings_from_raw.csv`. |

Every number printed by these scripts is the number quoted in the lessons. If yours differs, run `python data/make_dataset.py` again.
