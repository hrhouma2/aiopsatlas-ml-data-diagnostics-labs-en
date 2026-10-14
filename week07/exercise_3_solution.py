"""Week 7, Exercise 3 - A small neural network vs logistic regression on the fault task.

Run from the kit root, with the venv active:

    python week07/exercise_3_solution.py

Steps:
  1. load the Week 4 feature table (rebuilt here if data/features.csv is missing);
  2. temporal split, then scale with a scaler fitted on the training rows only;
  3. logistic regression, the reference;
  4. MLPClassifier(32, 16) with early stopping;
  5. read the training: epochs, loss curve, validation score; plot to week07/mlp_loss.png;
  6. the same network without early stopping, for comparison.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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


# Step 1 - the table
print("== Step 1: the feature table ==")
data = make_features()
print(data.shape)

# Step 2 - split, then scale
print()
print("== Step 2: temporal split and scaling ==")
train = data[data["date"] < "2025-10-01"]
test = data[data["date"] >= "2025-10-01"]
scaler = StandardScaler().fit(train[FEATURES])
X_train, y_train = scaler.transform(train[FEATURES]), train[TARGET]
X_test, y_test = scaler.transform(test[FEATURES]), test[TARGET]
print(f"train rows: {len(train)}  test rows: {len(test)}  features: {X_train.shape[1]}")

# Step 3 - the reference
print()
print("== Step 3: logistic regression ==")
logistic = LogisticRegression(max_iter=1000).fit(X_train, y_train)
auc_logistic = roc_auc_score(y_test, logistic.predict_proba(X_test)[:, 1])
print(f"logistic AUC (test): {auc_logistic:.3f}")

# Step 4 - the network
print()
print("== Step 4: MLPClassifier with early stopping ==")
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True, random_state=42, max_iter=300)
mlp.fit(X_train, y_train)
auc_mlp = roc_auc_score(y_test, mlp.predict_proba(X_test)[:, 1])
print(f"MLP AUC (test): {auc_mlp:.3f}")
weights = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
print(f"layers: {mlp.n_layers_}  weights and biases: {weights}")

# Step 5 - read the training
print()
print("== Step 5: what happened during training ==")
print(f"epochs run: {mlp.n_iter_}  (max_iter was 300)")
print(f"best validation score: {mlp.best_validation_score_:.3f}  (accuracy on the 10 % held out by early_stopping)")
print(f"loss after epoch 1: {mlp.loss_curve_[0]:.3f}  after the last epoch: {mlp.loss_curve_[-1]:.3f}")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].plot(range(1, mlp.n_iter_ + 1), mlp.loss_curve_)
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("training loss")
axes[0].set_title("Loss goes down")
axes[1].plot(range(1, mlp.n_iter_ + 1), mlp.validation_scores_)
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("validation accuracy")
axes[1].set_title("Validation score stops improving")
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("week07/mlp_loss.png", dpi=120)
print("saved week07/mlp_loss.png")

# Step 6 - without early stopping
print()
print("== Step 6: the same network without early stopping ==")
rows = []
for stop in [True, False]:
    model = MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=stop, random_state=42, max_iter=300).fit(X_train, y_train)
    rows.append(
        {
            "early_stopping": stop,
            "epochs": model.n_iter_,
            "auc_train": round(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]), 3),
            "auc_test": round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 3),
        }
    )
print(pd.DataFrame(rows).to_string(index=False))
print(f"logistic, for reference: auc_test {auc_logistic:.3f}")
