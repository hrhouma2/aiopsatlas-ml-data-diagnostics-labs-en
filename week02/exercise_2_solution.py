"""Week 2, Exercise 2 - Six SQL queries, and the same six in pandas.

Run from the kit root, with the venv active:

    python week02/exercise_2_solution.py

Every query runs twice: once in SQL on data/diagnostics.sqlite with sqlite3,
once in pandas on the CSV files of data/clean/. The script checks that the
two answers are the same.
"""

import sqlite3

import pandas as pd

pd.set_option("display.width", 120)

con = sqlite3.connect("data/diagnostics.sqlite")
machines = pd.read_csv("data/clean/machines.csv")
readings = pd.read_csv("data/clean/readings.csv")
incidents = pd.read_csv("data/clean/incidents.csv")


def sql(query):
    """Run one SELECT and return a DataFrame."""
    return pd.read_sql_query(query, con)


# Query 1 - COUNT
print("== Query 1: how many readings? ==")
q1 = sql("SELECT COUNT(*) AS n FROM readings")
print(q1)
p1 = len(readings)
print("pandas:", p1)
assert q1["n"][0] == p1
print()

# Query 2 - WHERE + ORDER BY + LIMIT
print("== Query 2: hot days of M001 ==")
q2 = sql(
    """
    SELECT date, temperature_c
    FROM readings
    WHERE machine_id = 'M001' AND temperature_c > 60
    ORDER BY date
    LIMIT 5
    """
)
print(q2)
mask = (readings["machine_id"] == "M001") & (readings["temperature_c"] > 60)
p2 = readings.loc[mask, ["date", "temperature_c"]].sort_values("date").head(5)
print(p2.to_string(index=False))
assert q2["date"].tolist() == p2["date"].tolist()
print()

# Query 3 - GROUP BY + COUNT
print("== Query 3: machines per site ==")
q3 = sql("SELECT site, COUNT(*) AS n FROM machines GROUP BY site ORDER BY n DESC")
print(q3)
p3 = machines["site"].value_counts()
print(p3)
assert dict(zip(q3["site"], q3["n"])) == p3.to_dict()
print()

# Query 4 - GROUP BY + AVG
print("== Query 4: the five hottest machines ==")
q4 = sql(
    """
    SELECT machine_id, ROUND(AVG(temperature_c), 1) AS mean_temp
    FROM readings
    GROUP BY machine_id
    ORDER BY mean_temp DESC
    LIMIT 5
    """
)
print(q4)
p4 = readings.groupby("machine_id")["temperature_c"].mean().round(1).sort_values(ascending=False).head(5)
print(p4)
assert q4["machine_id"].tolist() == p4.index.tolist()
print()

# Query 5 - JOIN
print("== Query 5: mean power by machine type ==")
q5 = sql(
    """
    SELECT m.machine_type, ROUND(AVG(r.power_kw), 1) AS mean_power
    FROM readings r
    JOIN machines m ON r.machine_id = m.machine_id
    GROUP BY m.machine_type
    ORDER BY mean_power DESC
    """
)
print(q5)
merged = readings.merge(machines, on="machine_id")
p5 = merged.groupby("machine_type")["power_kw"].mean().round(1).sort_values(ascending=False)
print(p5)
assert q5["mean_power"].tolist() == p5.tolist()
print()

# Query 6 - GROUP BY with two aggregates
print("== Query 6: incidents and cost by category ==")
q6 = sql(
    """
    SELECT category, COUNT(*) AS n, ROUND(SUM(repair_cost_cad), 2) AS cost
    FROM incidents
    GROUP BY category
    ORDER BY n DESC
    """
)
print(q6)
p6 = (
    incidents.groupby("category")
    .agg(n=("incident_id", "count"), cost=("repair_cost_cad", "sum"))
    .round(2)
    .sort_values("n", ascending=False)
)
print(p6)
assert q6["n"].tolist() == p6["n"].tolist()
assert q6["cost"].tolist() == p6["cost"].tolist()
print()

con.close()
print("All six queries give the same answer in SQL and in pandas.")
