"""Week 3, Exercise 3 - Graph traversal with pandas merges.

Run from the kit root, with the venv active:

    python week03/exercise_3_solution.py

Walks the NorthPeak knowledge graph (data/graph/nodes.csv, edges.csv) with
nothing but pandas merges:
  1. load nodes and edges, count them by label and by relation;
  2. the neighbours of one machine, M001;
  3. one hop per relation: four small tables;
  4. a 2-hop path machine -> incident -> category, for M001;
  5. a 3-hop path category <- incident -> machine -> site:
     which sites have the most bearing_wear incidents;
  6. the most connected node (degree).
"""

import pandas as pd

pd.set_option("display.width", 120)

# Step 1 - load
print("== Step 1: nodes and edges ==")
nodes = pd.read_csv("data/graph/nodes.csv")
edges = pd.read_csv("data/graph/edges.csv")
print("nodes:", len(nodes), "| edges:", len(edges))
print(nodes["label"].value_counts())
print(edges["relation"].value_counts())
print()

# Step 2 - neighbours of M001
print("== Step 2: neighbours of M001 ==")
print(edges[edges["source"] == "M001"])
print(edges[edges["target"] == "M001"])
print()

# Step 3 - one table per relation
print("== Step 3: one table per relation ==")


def relation(name, source_name, target_name):
    """Keep one relation and rename source/target to readable names."""
    part = edges[edges["relation"] == name]
    return part.rename(columns={"source": source_name, "target": target_name})[[source_name, target_name]]


located_at = relation("LOCATED_AT", "machine", "site")
is_a = relation("IS_A", "machine", "machine_type")
happened_on = relation("HAPPENED_ON", "incident", "machine")
has_category = relation("HAS_CATEGORY", "incident", "category")
print(happened_on.head(3))
print(has_category.head(3))
print()

# Step 4 - 2 hops: machine -> incident -> category
print("== Step 4: 2-hop path for M001 ==")
two_hop = happened_on.merge(has_category, on="incident")
print(two_hop[two_hop["machine"] == "M001"])
print()

# Step 5 - 3 hops: category <- incident -> machine -> site
print("== Step 5: bearing_wear incidents by site ==")
three_hop = two_hop.merge(located_at, on="machine")
print("rows:", len(three_hop))
bearing = three_hop[three_hop["category"] == "bearing_wear"]
print(bearing["site"].value_counts())
print()
print(pd.crosstab(three_hop["site"], three_hop["category"]))
print()

# Step 6 - degree
print("== Step 6: the most connected nodes ==")
degree = pd.concat([edges["source"], edges["target"]]).value_counts()
machine_degree = degree[degree.index.isin(located_at["machine"])]
print(machine_degree.head(3))
top = machine_degree.index[0]
print(f"{top}: {int(happened_on['machine'].eq(top).sum())} incidents + LOCATED_AT + IS_A = {int(machine_degree.iloc[0])} edges")
