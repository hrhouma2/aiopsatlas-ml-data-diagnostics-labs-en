"""Week 5, Exercise 2 - Predict fault_next_7d with the Week 4 features.

Run from the kit root, with the venv active, after Week 4 Exercise 1 (data/features.csv exists):

    python week05/exercise_2_solution.py

  1. temporal split, 15 feature columns;
  2. DummyClassifier (always 0): accuracy 0.936, which means nothing;
  3. Pipeline(StandardScaler, LogisticRegression): probabilities, AUC;
  4. confusion matrix at threshold 0.5;
  5. precision, recall, F1, accuracy;
  6. the coefficients, sorted.
"""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

features = pd.read_csv("data/features.csv")
FEATURES = [c for c in features.columns if c not in ("machine_id", "date", "fault_next_7d")]

# Step 1 - split
print("== Step 1: temporal split ==")
train = features[features["date"] < "2025-10-01"]
test = features[features["date"] >= "2025-10-01"]
X_train, y_train = train[FEATURES], train["fault_next_7d"]
X_test, y_test = test[FEATURES], test["fault_next_7d"]
print(len(FEATURES), "features:", FEATURES)
print("train", X_train.shape, " test", X_test.shape, " faults in test:", int(y_test.sum()))
print()

# Step 2 - baseline
print("== Step 2: the baseline, always 0 ==")
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print(f"dummy accuracy: {accuracy_score(y_test, dummy.predict(X_test)):.4f}")
print(f"dummy recall  : {recall_score(y_test, dummy.predict(X_test)):.4f}")
print()

# Step 3 - logistic regression, probabilities
print("== Step 3: logistic regression, a probability per row ==")
pipe = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))])
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
print("first 5 probabilities:", proba[:5].round(3))
print(f"AUC: {roc_auc_score(y_test, proba):.3f}")
print()

# Step 4 - confusion matrix at 0.5
print("== Step 4: confusion matrix at threshold 0.5 ==")
pred = (proba >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print(confusion_matrix(y_test, pred))
print(f"TN {tn}  FP {fp}  FN {fn}  TP {tp}")
print()

# Step 5 - the four scores
print("== Step 5: accuracy, precision, recall, F1 ==")
print(f"accuracy  {accuracy_score(y_test, pred):.4f}")
print(f"precision {precision_score(y_test, pred):.3f}")
print(f"recall    {recall_score(y_test, pred):.3f}")
print(f"F1        {f1_score(y_test, pred):.3f}")
print()

# Step 6 - coefficients
print("== Step 6: the coefficients, sorted ==")
coefficients = pd.Series(pipe.named_steps["model"].coef_[0], index=FEATURES).round(3).sort_values()
print(coefficients.to_string())
