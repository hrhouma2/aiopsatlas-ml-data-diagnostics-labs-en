"""Week 4, Exercise 3 - Leakage: see it, then make it impossible.

Run from the kit root, with the venv active, after Exercise 1 (data/features.csv exists):

    python week04/exercise_3_solution.py

  1. temporal split of data/features.csv;
  2. a scaler fitted on ALL rows sees the test months (its mean changes);
  3. the AUC does not move here: leakage can be silent;
  4. a column that knows the future (days_to_incident) gives an AUC near 1;
  5. a Pipeline fitted on train only cannot see the test rows.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

features = pd.read_csv("data/features.csv")
FEATURES = [c for c in features.columns if c not in ("machine_id", "date", "fault_next_7d")]

# Step 1 - temporal split
print("== Step 1: temporal split ==")
train = features[features["date"] < "2025-10-01"]
test = features[features["date"] >= "2025-10-01"]
X_train, y_train = train[FEATURES], train["fault_next_7d"]
X_test, y_test = test[FEATURES], test["fault_next_7d"]
print("train", X_train.shape, " test", X_test.shape)
print()

# Step 2 - a scaler fitted before the split
print("== Step 2: scaler fitted on all rows vs on train only ==")
scaler_all = StandardScaler().fit(features[FEATURES])
scaler_train = StandardScaler().fit(X_train)
i = FEATURES.index("ambient_c")
print(f"mean ambient_c seen by the scaler: all rows {scaler_all.mean_[i]:.3f}   train only {scaler_train.mean_[i]:.3f}")
first = X_test.iloc[[0]]
print("first test row ambient_c:", float(first["ambient_c"].iloc[0]))
print(f"scaled with all rows  : {scaler_all.transform(first)[0, i]:.3f}")
print(f"scaled with train only: {scaler_train.transform(first)[0, i]:.3f}")
print()

# Step 3 - does the AUC move?
print("== Step 3: AUC with the leaky scaler vs the clean one ==")
model_leaky = LogisticRegression(max_iter=2000).fit(scaler_all.transform(X_train), y_train)
model_clean = LogisticRegression(max_iter=2000).fit(scaler_train.transform(X_train), y_train)
auc_leaky = roc_auc_score(y_test, model_leaky.predict_proba(scaler_all.transform(X_test))[:, 1])
auc_clean = roc_auc_score(y_test, model_clean.predict_proba(scaler_train.transform(X_test))[:, 1])
print(f"AUC leaky scaler {auc_leaky:.4f}   AUC clean scaler {auc_clean:.4f}")
print()

# Step 4 - a column that knows the future
print("== Step 4: a column that knows the future ==")
incidents = pd.read_csv("data/clean/incidents.csv")
left = features.assign(date_dt=pd.to_datetime(features["date"])).sort_values("date_dt")
right = incidents.assign(incident_dt=pd.to_datetime(incidents["date"])).sort_values("incident_dt")
leaky = pd.merge_asof(left, right[["machine_id", "incident_dt"]], left_on="date_dt", right_on="incident_dt",
                      by="machine_id", direction="forward")
leaky["days_to_incident"] = (leaky["incident_dt"] - leaky["date_dt"]).dt.days.fillna(999)
leaky = leaky.sort_values(["machine_id", "date"])
print(leaky[leaky["machine_id"] == "M001"][["date", "days_to_incident", "fault_next_7d"]].iloc[38:46].to_string(index=False))
train_l = leaky[leaky["date"] < "2025-10-01"]
test_l = leaky[leaky["date"] >= "2025-10-01"]
cols = FEATURES + ["days_to_incident"]
pipe_leak = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))])
pipe_leak.fit(train_l[cols], train_l["fault_next_7d"])
print(f"AUC with days_to_incident: {roc_auc_score(test_l['fault_next_7d'], pipe_leak.predict_proba(test_l[cols])[:, 1]):.3f}")
print()

# Step 5 - the pipeline that makes scaler leakage impossible
print("== Step 5: Pipeline fitted on train only ==")
pipe = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))])
pipe.fit(X_train, y_train)
print(f"mean ambient_c inside the pipeline scaler: {pipe.named_steps['scale'].mean_[i]:.3f}")
print(f"AUC pipeline: {roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]):.3f}")
