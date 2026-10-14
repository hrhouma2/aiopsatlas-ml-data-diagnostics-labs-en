"""Week 3, Exercise 1 - Three-table join, in SQL and in pandas.

Run from the kit root, with the venv active:

    python week03/exercise_1_solution.py

Joins incidents, machines and readings on data/diagnostics.sqlite, then does
the same joins with pandas merge on the CSV files, and checks that the
numbers are identical:
  1. incidents JOIN machines: count and cost by site;
  2. the same by site and machine type;
  3. + readings of the incident day: mean temperature that day;
  4. pandas merge, two tables;
  5. pandas merge, three tables;
  6. assert: SQL == pandas.
"""

import sqlite3

import pandas as pd

pd.set_option("display.width", 120)

con = sqlite3.connect("data/diagnostics.sqlite")
machines = pd.read_csv("data/clean/machines.csv")
readings = pd.read_csv("data/clean/readings.csv")
incidents = pd.read_csv("data/clean/incidents.csv")

# Step 1 - two tables: incidents by site
print("== Step 1: SQL, incidents by site ==")
sql_site = pd.read_sql_query(
    """
    SELECT m.site, COUNT(*) AS n, ROUND(SUM(i.repair_cost_cad), 2) AS cost
    FROM incidents i
    JOIN machines m ON i.machine_id = m.machine_id
    GROUP BY m.site
    ORDER BY n DESC
    """,
    con,
)
print(sql_site)
print()

# Step 2 - two tables, two group keys
print("== Step 2: SQL, incidents by site and type ==")
sql_site_type = pd.read_sql_query(
    """
    SELECT m.site, m.machine_type, COUNT(*) AS n, ROUND(SUM(i.repair_cost_cad), 2) AS cost
    FROM incidents i
    JOIN machines m ON i.machine_id = m.machine_id
    GROUP BY m.site, m.machine_type
    ORDER BY n DESC
    """,
    con,
)
print(sql_site_type)
print()

# Step 3 - three tables: add the reading of the incident day
print("== Step 3: SQL, three tables ==")
sql_three = pd.read_sql_query(
    """
    SELECT m.site, m.machine_type, COUNT(*) AS n, ROUND(AVG(r.temperature_c), 1) AS temp_that_day
    FROM incidents i
    JOIN machines m ON i.machine_id = m.machine_id
    JOIN readings r ON r.machine_id = i.machine_id AND r.date = i.date
    GROUP BY m.site, m.machine_type
    ORDER BY n DESC
    """,
    con,
)
print(sql_three)
print()

# Step 4 - pandas, two tables
print("== Step 4: pandas, incidents by site ==")
two = incidents.merge(machines, on="machine_id")
print("merged rows:", len(two))
pd_site = (
    two.groupby("site")
    .agg(n=("incident_id", "count"), cost=("repair_cost_cad", "sum"))
    .round(2)
    .sort_values("n", ascending=False)
    .reset_index()
)
print(pd_site)
pd_site_type = (
    two.groupby(["site", "machine_type"])
    .agg(n=("incident_id", "count"), cost=("repair_cost_cad", "sum"))
    .round(2)
    .sort_values("n", ascending=False)
    .reset_index()
)
print()

# Step 5 - pandas, three tables
print("== Step 5: pandas, three tables ==")
three = two.merge(readings, on=["machine_id", "date"])
print("merged rows:", len(three))
pd_three = (
    three.groupby(["site", "machine_type"])
    .agg(n=("incident_id", "count"), temp_that_day=("temperature_c", "mean"))
    .round(1)
    .sort_values("n", ascending=False)
    .reset_index()
)
print(pd_three)
print()

# Step 6 - same numbers?
print("== Step 6: SQL == pandas ==")
assert sql_site["n"].tolist() == pd_site["n"].tolist()
assert sql_site["cost"].tolist() == pd_site["cost"].tolist()
print("by site          : same counts, same costs")
sql_st = sql_site_type.set_index(["site", "machine_type"]).sort_index()
pd_st = pd_site_type.set_index(["site", "machine_type"]).sort_index()
assert sql_st["n"].tolist() == pd_st["n"].tolist()
assert sql_st["cost"].tolist() == pd_st["cost"].tolist()
print("by site and type : same counts, same costs")
sql_t = sql_three.set_index(["site", "machine_type"]).sort_index()
pd_t = pd_three.set_index(["site", "machine_type"]).sort_index()
assert sql_t["n"].tolist() == pd_t["n"].tolist()
# SQLite and Python can round a value ending in .x5 in different directions.
gap = (sql_t["temp_that_day"] - pd_t["temp_that_day"]).abs().max()
assert gap < 0.15, gap
print(f"three tables     : same counts, mean temperatures within {gap:.1f} (rounding of .x5)")
print("total incidents  :", int(sql_site["n"].sum()), "| total cost:", round(sql_site["cost"].sum(), 2))
con.close()
