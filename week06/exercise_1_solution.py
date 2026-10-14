"""Week 6, Exercise 1 - Tree depth sweep: train vs validation curve.

Run from the kit root, with the venv active:

    python week06/exercise_1_solution.py

Steps:
  1. load the Week 4 feature table (rebuilt here if data/features.csv is missing);
  2. cut the training months in two: fit on Jan-Jun, validate on Jul-Sep;
  3. read the rules of a depth-2 tree;
  4. sweep max_depth from 1 to 20 and record the AUC on both parts;
  5. plot the two curves to week06/depth_sweep.png and pick the best depth.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

FEATURES = [
    "load_pct", "ambient_c", "temperature_c", "vibration_mm_s", "pressure_bar", "machine_age",
    "temperature_c_delta30", "vibration_mm_s_delta30", "temperature_c_trend", "vibration_mm_s_trend",
    "type_pump", "type_compressor", "type_conveyor", "type_chiller",
]
TARGET = "fault_next_7d"


def make_features(path="data/features.csv"):
    """Return the Week 4 feature table. Reads data/features.csv when it exists, rebuilds it otherwise."""
    if os.path.exists(path):
        table = pd.read_csv(path)
        if set(FEATURES + [TARGET, "machine_id", "date"]).issubset(table.columns):
            return table
    readings = pd.read_csv("data/clean/readings.csv")
    machines = pd.read_csv("data/clean/machines.csv")
    table = readings.merge(machines, on="machine_id").sort_values(["machine_id", "date"]).reset_index(drop=True)
    table["machine_age"] = 2025 - table["install_year"]
    by_machine = table.groupby("machine_id")
    for col in ["temperature_c", "vibration_mm_s"]:
        table[f"{col}_delta30"] = table[col] - by_machine[col].transform(
            lambda s: s.rolling(30, min_periods=5).mean().shift(1)
        )
        table[f"{col}_trend"] = by_machine[col].transform(
            lambda s: s.rolling(3, min_periods=1).mean() - s.rolling(14, min_periods=3).mean()
        )
    for kind in ["pump", "compressor", "conveyor", "chiller"]:
        table[f"type_{kind}"] = (table["machine_type"] == kind).astype(int)
    return table.dropna().reset_index(drop=True)


# Step 1 - the feature table
print("== Step 1: the feature table ==")
data = make_features()
print(data.shape)
print(data[["machine_id", "date", "temperature_c", "temperature_c_delta30", "vibration_mm_s_trend", TARGET]].head().round(2).to_string(index=False))

# Step 2 - fit part and validation part, both before October
print()
print("== Step 2: fit on Jan-Jun, validate on Jul-Sep ==")
train = data[data["date"] < "2025-10-01"]
fit = train[train["date"] < "2025-07-01"]
val = train[train["date"] >= "2025-07-01"]
X_fit, y_fit = fit[FEATURES], fit[TARGET]
X_val, y_val = val[FEATURES], val[TARGET]
print(f"fit rows: {len(fit)}  positives: {y_fit.mean():.3f}")
print(f"val rows: {len(val)}  positives: {y_val.mean():.3f}")

# Step 3 - a depth-2 tree you can read
print()
print("== Step 3: the rules of a depth-2 tree ==")
small = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X_fit, y_fit)
print(export_text(small, feature_names=FEATURES, decimals=2))
print(f"AUC fit: {roc_auc_score(y_fit, small.predict_proba(X_fit)[:, 1]):.3f}")
print(f"AUC val: {roc_auc_score(y_val, small.predict_proba(X_val)[:, 1]):.3f}")

# Step 4 - the sweep
print()
print("== Step 4: depth 1 to 20 ==")
rows = []
for depth in range(1, 21):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_fit, y_fit)
    rows.append(
        {
            "depth": depth,
            "leaves": tree.get_n_leaves(),
            "auc_fit": roc_auc_score(y_fit, tree.predict_proba(X_fit)[:, 1]),
            "auc_val": roc_auc_score(y_val, tree.predict_proba(X_val)[:, 1]),
        }
    )
sweep = pd.DataFrame(rows)
print(sweep.round(3).to_string(index=False))

# Step 5 - plot and pick
print()
print("== Step 5: the curve ==")
best = sweep.loc[sweep["auc_val"].idxmax()]
plt.figure(figsize=(7, 4))
plt.plot(sweep["depth"], sweep["auc_fit"], marker="o", label="fit (Jan-Jun)")
plt.plot(sweep["depth"], sweep["auc_val"], marker="s", label="validation (Jul-Sep)")
plt.axvline(best["depth"], color="grey", linestyle="--", label=f"best depth = {int(best['depth'])}")
plt.xlabel("max_depth")
plt.ylabel("AUC")
plt.title("Decision tree: deeper is not better")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("week06/depth_sweep.png", dpi=120)
print(f"best depth: {int(best['depth'])}  AUC val: {best['auc_val']:.3f}  AUC fit: {best['auc_fit']:.3f}")
last = sweep.iloc[-1]
print(f"depth 20   AUC val: {last['auc_val']:.3f}  AUC fit: {last['auc_fit']:.3f}  leaves: {int(last['leaves'])}")
print("saved week06/depth_sweep.png")
