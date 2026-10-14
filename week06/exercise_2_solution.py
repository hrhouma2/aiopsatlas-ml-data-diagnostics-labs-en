"""Week 6, Exercise 2 - Grid search on a random forest with temporal cross-validation.

Run from the kit root, with the venv active:

    python week06/exercise_2_solution.py

Steps:
  1. load the Week 4 feature table (rebuilt here if data/features.csv is missing);
  2. keep the training months (before 2025-10-01) and look at the TimeSeriesSplit folds;
  3. score a default random forest with that split;
  4. run a small GridSearchCV (12 settings, 3 folds);
  5. read the results table, save the best settings to week06/best_params.json.
"""

import json
import os
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score

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


# Step 1 - the training rows, sorted by date
print("== Step 1: training rows ==")
data = make_features()
train = data[data["date"] < "2025-10-01"].sort_values(["date", "machine_id"]).reset_index(drop=True)
X, y = train[FEATURES], train[TARGET]
print(f"train rows: {len(train)}  from {train['date'].min()} to {train['date'].max()}")

# Step 2 - the folds of TimeSeriesSplit
print()
print("== Step 2: TimeSeriesSplit folds ==")
tscv = TimeSeriesSplit(n_splits=3)
for k, (fit_idx, val_idx) in enumerate(tscv.split(X), start=1):
    fit_dates, val_dates = train.loc[fit_idx, "date"], train.loc[val_idx, "date"]
    print(
        f"fold {k}: fit {fit_dates.min()} -> {fit_dates.max()} ({len(fit_idx)} rows) | "
        f"validate {val_dates.min()} -> {val_dates.max()} ({len(val_idx)} rows)"
    )

# Step 3 - a default forest, scored with the temporal folds
print()
print("== Step 3: default random forest ==")
default = RandomForestClassifier(random_state=42, n_jobs=-1)
scores = cross_val_score(default, X, y, cv=tscv, scoring="roc_auc")
print("AUC per fold:", scores.round(3))
print(f"mean AUC: {scores.mean():.3f}")

# Step 4 - a small grid
print()
print("== Step 4: grid search ==")
grid = {
    "n_estimators": [100, 300],
    "max_depth": [4, 8, None],
    "min_samples_leaf": [1, 20],
}
search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=grid,
    cv=tscv,
    scoring="roc_auc",
    return_train_score=True,
)
start = time.perf_counter()
search.fit(X, y)
print(f"12 settings x 3 folds = {len(search.cv_results_['params']) * tscv.n_splits} fits in {time.perf_counter() - start:.0f} s")

# Step 5 - read the table, save the best
print()
print("== Step 5: results ==")
results = pd.DataFrame(search.cv_results_)
table = results[
    ["param_n_estimators", "param_max_depth", "param_min_samples_leaf", "mean_train_score", "mean_test_score", "std_test_score", "rank_test_score"]
].sort_values("rank_test_score")
table.columns = ["n_estimators", "max_depth", "min_samples_leaf", "auc_train", "auc_val", "std_val", "rank"]
print(table.round(3).to_string(index=False))
print()
print("best params:", search.best_params_)
print(f"best mean AUC: {search.best_score_:.3f}")
with open("week06/best_params.json", "w", encoding="utf-8") as f:
    json.dump(search.best_params_, f, indent=2)
print("saved week06/best_params.json")
