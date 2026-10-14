"""Week 4, Exercise 2 - The three splits and their positive rates.

Run from the kit root, with the venv active, after Exercise 1 (data/features.csv exists):

    python week04/exercise_2_solution.py

Compares four ways to split data/features.csv into train and test:
  random with stratify, random without stratify, temporal (date), group (machine).
For each one: row counts, share of fault_next_7d = 1, and how many test rows
have their previous day inside the train set (the "neighbour" count).
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

features = pd.read_csv("data/features.csv")
y = features["fault_next_7d"]


def neighbours(test_df, train_df):
    """How many test rows have the same machine, one day earlier, in train?"""
    in_train = set(zip(train_df["machine_id"], train_df["date"]))
    previous = (pd.to_datetime(test_df["date"]) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    return sum((m, d) in in_train for m, d in zip(test_df["machine_id"], previous))


def report(name, train_df, test_df):
    print(f"{name:22s} train {len(train_df):>5}  test {len(test_df):>5}  "
          f"positive rate train {train_df['fault_next_7d'].mean():.4f}  test {test_df['fault_next_7d'].mean():.4f}  "
          f"neighbours {neighbours(test_df, train_df)}")


# Step 1 - the whole table
print("== Step 1: the whole table ==")
print("rows:", len(features), " positive rate:", round(y.mean(), 4))
print()

# Step 2 - random split, stratified
print("== Step 2: random split, stratified ==")
train_s, test_s = train_test_split(features, test_size=0.25, random_state=42, stratify=y)
report("random stratified", train_s, test_s)
print()

# Step 3 - random split, not stratified
print("== Step 3: random split, not stratified ==")
train_r, test_r = train_test_split(features, test_size=0.25, random_state=42)
report("random plain", train_r, test_r)
print()

# Step 4 - temporal split
print("== Step 4: temporal split ==")
train_t = features[features["date"] < "2025-10-01"]
test_t = features[features["date"] >= "2025-10-01"]
report("temporal", train_t, test_t)
print("train dates:", train_t["date"].min(), "to", train_t["date"].max())
print("test dates :", test_t["date"].min(), "to", test_t["date"].max())
print()

# Step 5 - group split by machine
print("== Step 5: group split by machine ==")
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(features, y, groups=features["machine_id"]))
train_g, test_g = features.iloc[train_idx], features.iloc[test_idx]
report("group by machine", train_g, test_g)
print("test machines:", sorted(test_g["machine_id"].unique()))
print()

# Step 6 - all together
print("== Step 6: the four splits side by side ==")
report("random stratified", train_s, test_s)
report("random plain", train_r, test_r)
report("temporal", train_t, test_t)
report("group by machine", train_g, test_g)
