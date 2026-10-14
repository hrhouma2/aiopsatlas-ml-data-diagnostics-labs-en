"""Week 5, Exercise 3 - Move the threshold from 0.5 to 0.2.

Run from the kit root, with the venv active, after Week 4 Exercise 1 (data/features.csv exists):

    python week05/exercise_3_solution.py

  1. the same model as Exercise 2, probabilities on the test quarter;
  2. a precision-recall table for thresholds 0.5, 0.4, 0.3, 0.2, 0.1;
  3. class_weight='balanced' at threshold 0.5;
  4. a cost per threshold: a missed fault costs 1,200 CAD, a false alarm 150 CAD;
  5. two plots saved to week05/threshold_curves.png and week05/precision_recall.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MISSED_FAULT_CAD = 1200
FALSE_ALARM_CAD = 150

features = pd.read_csv("data/features.csv")
FEATURES = [c for c in features.columns if c not in ("machine_id", "date", "fault_next_7d")]
train = features[features["date"] < "2025-10-01"]
test = features[features["date"] >= "2025-10-01"]
X_train, y_train = train[FEATURES], train["fault_next_7d"]
X_test, y_test = test[FEATURES], test["fault_next_7d"]


def row(name, y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    cost = fn * MISSED_FAULT_CAD + fp * FALSE_ALARM_CAD
    print(f"{name:10s} TN {tn:5d}  FP {fp:4d}  FN {fn:3d}  TP {tp:3d}   "
          f"precision {precision_score(y_true, pred):.3f}  recall {recall_score(y_true, pred):.3f}  "
          f"F1 {f1_score(y_true, pred):.3f}   cost {cost:>7,} CAD")


# Step 1 - the model and its probabilities
print("== Step 1: the Exercise 2 model again ==")
pipe = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))])
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
print(f"AUC {roc_auc_score(y_test, proba):.3f}   faults in test: {int(y_test.sum())}   test rows: {len(y_test)}")
print()

# Step 2 - thresholds
print("== Step 2: move the threshold ==")
for threshold in [0.5, 0.4, 0.3, 0.2, 0.1]:
    row(f"thr {threshold}", y_test, (proba >= threshold).astype(int))
print()

# Step 3 - class_weight balanced
print("== Step 3: class_weight='balanced' at 0.5 ==")
balanced = Pipeline([("scale", StandardScaler()),
                     ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])
balanced.fit(X_train, y_train)
proba_balanced = balanced.predict_proba(X_test)[:, 1]
row("balanced", y_test, (proba_balanced >= 0.5).astype(int))
print(f"AUC balanced {roc_auc_score(y_test, proba_balanced):.3f}")
print()

# Step 4 - the cheapest threshold
print("== Step 4: the cheapest threshold between 0.05 and 0.95 ==")
grid = [round(t, 2) for t in [x / 100 for x in range(5, 96, 5)]]
costs = []
for threshold in grid:
    tn, fp, fn, tp = confusion_matrix(y_test, (proba >= threshold).astype(int)).ravel()
    costs.append(fn * MISSED_FAULT_CAD + fp * FALSE_ALARM_CAD)
best = grid[costs.index(min(costs))]
print(f"cheapest threshold: {best}   cost {min(costs):,} CAD   (at 0.5: {costs[grid.index(0.5)]:,} CAD)")
print()

# Step 5 - plots
print("== Step 5: plots ==")
precisions, recalls, thresholds = [], [], [round(x / 100, 2) for x in range(2, 99, 2)]
for threshold in thresholds:
    pred = (proba >= threshold).astype(int)
    precisions.append(precision_score(y_test, pred, zero_division=0))
    recalls.append(recall_score(y_test, pred))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(thresholds, precisions, label="precision")
ax.plot(thresholds, recalls, label="recall")
ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
ax.axvline(0.2, color="grey", linestyle=":", linewidth=1)
ax.set_xlabel("threshold")
ax.set_ylabel("score")
ax.set_title("Precision and recall against the threshold")
ax.legend()
fig.tight_layout()
fig.savefig("week05/threshold_curves.png", dpi=120)

p, r, _ = precision_recall_curve(y_test, proba)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(r, p)
ax.set_xlabel("recall")
ax.set_ylabel("precision")
ax.set_title("Precision-recall curve, test quarter")
fig.tight_layout()
fig.savefig("week05/precision_recall.png", dpi=120)
print("saved week05/threshold_curves.png and week05/precision_recall.png")
