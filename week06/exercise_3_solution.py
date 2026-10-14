"""Week 6, Exercise 3 - Error analysis and the final test, used once.

Run from the kit root, with the venv active:

    python week06/exercise_3_solution.py

Steps:
  1. load the Week 4 feature table (rebuilt here if data/features.csv is missing);
  2. train on Jan-Sep, keep Oct-Dec as the test set, touched once;
  3. fit the tuned forest (week06/best_params.json) and a logistic regression;
  4. AUC and confusion matrix on the test set;
  5. feature importance of the forest;
  6. error analysis by machine type and by month;
  7. write week06/results.md.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "load_pct", "ambient_c", "temperature_c", "vibration_mm_s", "pressure_bar", "machine_age",
    "temperature_c_delta30", "vibration_mm_s_delta30", "temperature_c_trend", "vibration_mm_s_trend",
    "type_pump", "type_compressor", "type_conveyor", "type_chiller",
]
TARGET = "fault_next_7d"
THRESHOLD = 0.3


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


def load_best_params(path="week06/best_params.json"):
    """Best forest settings from Exercise 2, with the values we found as a fallback."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"max_depth": None, "min_samples_leaf": 20, "n_estimators": 300}


# Step 1 - the table, plus machine_type for the error analysis
print("== Step 1: the feature table ==")
data = make_features()
machines = pd.read_csv("data/clean/machines.csv")[["machine_id", "machine_type"]]
data = data.merge(machines, on="machine_id", how="left")
print(data.shape)

# Step 2 - train and test
print()
print("== Step 2: train Jan-Sep, test Oct-Dec ==")
train = data[data["date"] < "2025-10-01"]
test = data[data["date"] >= "2025-10-01"].copy()
X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]
print(f"train rows: {len(train)}  positives: {y_train.mean():.3f}")
print(f"test rows : {len(test)}  positives: {y_test.mean():.3f}  faults: {int(y_test.sum())}")

# Step 3 - two models, fitted on the whole training set
print()
print("== Step 3: fit the tuned forest and a logistic regression ==")
params = load_best_params()
print("forest params:", params)
forest = RandomForestClassifier(random_state=42, n_jobs=-1, **params).fit(X_train, y_train)
logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
test["p_forest"] = forest.predict_proba(X_test)[:, 1]
test["p_logistic"] = logistic.predict_proba(X_test)[:, 1]

# Step 4 - the final numbers
print()
print("== Step 4: AUC and confusion matrix on the test set ==")
auc = {}
for name in ["forest", "logistic"]:
    auc[name] = roc_auc_score(y_test, test[f"p_{name}"])
    predicted = (test[f"p_{name}"] >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    print(f"{name:9s} AUC {auc[name]:.3f} | threshold {THRESHOLD}: TP {tp}  FP {fp}  FN {fn}  TN {tn} | recall {tp / (tp + fn):.2f}  precision {tp / (tp + fp):.2f}")

# Step 5 - feature importance
print()
print("== Step 5: feature importance of the forest ==")
importance = pd.Series(forest.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(importance.round(3).to_string())
plt.figure(figsize=(7, 4.5))
importance.sort_values().plot.barh()
plt.xlabel("importance (share of the splits' gain)")
plt.title("Random forest: which columns matter")
plt.tight_layout()
plt.savefig("week06/feature_importance.png", dpi=120)
print("saved week06/feature_importance.png")

# Step 6 - error analysis
print()
print("== Step 6: errors by machine type and by month ==")
test["hit"] = (test["p_forest"] >= THRESHOLD).astype(int)
test["month"] = test["date"].str[:7]


def to_markdown(frame):
    """A Markdown table from a DataFrame, without any extra package."""
    header = "| " + " | ".join(frame.columns) + " |"
    sep = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, sep] + body)


def group_report(frame, key):
    rows = []
    for value, part in frame.groupby(key):
        rows.append(
            {
                key: value,
                "rows": len(part),
                "faults": int(part[TARGET].sum()),
                "auc": roc_auc_score(part[TARGET], part["p_forest"]),
                "recall": recall_score(part[TARGET], part["hit"]),
                "false_alarms": int(((part["hit"] == 1) & (part[TARGET] == 0)).sum()),
            }
        )
    return pd.DataFrame(rows)


by_type = group_report(test, "machine_type")
by_month = group_report(test, "month")
print(by_type.round(3).to_string(index=False))
print()
print(by_month.round(3).to_string(index=False))

# Step 7 - results.md
print()
print("== Step 7: write week06/results.md ==")
lines = [
    "# Week 6 - Final results on the held-out last quarter",
    "",
    "Trained on 2025-01-06 to 2025-09-30, tested once on 2025-10-01 to 2025-12-31.",
    "",
    f"- Test rows: {len(test)}, faults: {int(y_test.sum())} ({y_test.mean():.1%})",
    f"- Forest settings (Exercise 2): `{params}`",
    f"- Decision threshold: {THRESHOLD}",
    "",
    "## Scores",
    "",
    "| Model | AUC (test) |",
    "| --- | --- |",
    f"| Random forest (tuned) | {auc['forest']:.3f} |",
    f"| Logistic regression | {auc['logistic']:.3f} |",
    "",
    "## Top 5 features (forest importance)",
    "",
]
for name, value in importance.head(5).items():
    lines.append(f"- `{name}`: {value:.3f}")
lines += ["", "## By machine type (forest)", "", to_markdown(by_type.round(3)), ""]
lines += ["## By month (forest)", "", to_markdown(by_month.round(3)), ""]
with open("week06/results.md", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines) + "\n")
print("saved week06/results.md")
