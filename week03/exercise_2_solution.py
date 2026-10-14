"""Week 3, Exercise 2 - JSON documents with the json module and DuckDB.

Run from the kit root, with the venv active:

    python week03/exercise_2_solution.py

Reads data/json/incidents.json two ways:
  1. json.load: a list of 155 dicts, nested access with ["machine"]["site"];
  2. a count by site with a loop;
  3. a tag filter in Python: "high" and "pump";
  4. DuckDB read_json_auto: the same file as a table, nested columns;
  5. DuckDB: the same count by site and the same tag filter;
  6. a small dict cache in front of a slow function.
"""

import json
import time
from collections import Counter

import duckdb

# Step 1 - load the documents
print("== Step 1: json.load ==")
with open("data/json/incidents.json", encoding="utf-8") as f:
    docs = json.load(f)
print("documents:", len(docs))
first = docs[0]
print("keys:", list(first.keys()))
print("first machine:", first["machine"])
print("site of the first incident:", first["machine"]["site"])
print("tags:", first["tags"])
print()

# Step 2 - count by site, nested field
print("== Step 2: incidents by site ==")
by_site = Counter(doc["machine"]["site"] for doc in docs)
for site, n in by_site.most_common():
    print(f"{site:<12} {n}")
print()

# Step 3 - tag filter in Python
print("== Step 3: tag filter ==")
high = [doc for doc in docs if "high" in doc["tags"]]
print("tag 'high'          :", len(high))
high_pump = [doc["_id"] for doc in docs if "high" in doc["tags"] and "pump" in doc["tags"]]
print("tags 'high' + 'pump':", len(high_pump), high_pump)
print()

# Step 4 - DuckDB reads the same file as a table
print("== Step 4: DuckDB read_json_auto ==")
print(duckdb.sql("SELECT COUNT(*) AS n FROM read_json_auto('data/json/incidents.json')"))
print(
    duckdb.sql(
        """
        SELECT column_name, column_type
        FROM (DESCRIBE SELECT * FROM read_json_auto('data/json/incidents.json'))
        WHERE column_name IN ('machine', 'tags', 'date')
        """
    )
)

# Step 5 - the same two questions in SQL
print("== Step 5: DuckDB, nested field and list ==")
print(
    duckdb.sql(
        """
        SELECT machine.site AS site, COUNT(*) AS n
        FROM read_json_auto('data/json/incidents.json')
        GROUP BY machine.site
        ORDER BY n DESC
        """
    )
)
print(
    duckdb.sql(
        """
        SELECT _id, machine.id AS machine_id, tags
        FROM read_json_auto('data/json/incidents.json')
        WHERE list_contains(tags, 'high') AND list_contains(tags, 'pump')
        ORDER BY _id
        """
    )
)

# Step 6 - a dict as a cache
print("== Step 6: a dict cache ==")
cache = {}


def count_by_site(site):
    """Slow on purpose: reads the whole file every time."""
    if site in cache:
        return cache[site], "from cache"
    time.sleep(0.5)
    with open("data/json/incidents.json", encoding="utf-8") as f:
        n = sum(1 for doc in json.load(f) if doc["machine"]["site"] == site)
    cache[site] = n
    return n, "computed"


for _ in range(2):
    start = time.perf_counter()
    n, how = count_by_site("Toronto")
    print(f"Toronto -> {n} ({how}, {time.perf_counter() - start:.3f} s)")
print("cache keys:", list(cache.keys()))
