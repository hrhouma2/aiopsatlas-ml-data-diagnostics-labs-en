"""Week 5, Exercise 1 - Predict power_kw and beat the baseline.

Run from the kit root, with the venv active, after Week 4 Exercise 1 (data/features.csv exists):

    python week05/exercise_1_solution.py

  1. temporal split, X = load_pct, ambient_c, type_*; y = power_kw;
  2. DummyRegressor (always the mean): MAE, RMSE, R2;
  3. LinearRegression on load_pct alone;
  4. LinearRegression on all six columns;
  5. read the coefficients;
  6. error by machine type and a scatter plot saved to week05/power_actual_vs_predicted.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

features = pd.read_csv("data/features.csv")
COLUMNS = ["load_pct", "ambient_c", "type_pump", "type_compressor", "type_conveyor", "type_chiller"]


def scores(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name:26s} MAE {mae:6.2f}   RMSE {rmse:6.2f}   R2 {r2:6.3f}")


# Step 1 - split
print("== Step 1: temporal split ==")
train = features[features["date"] < "2025-10-01"]
test = features[features["date"] >= "2025-10-01"]
X_train, y_train = train[COLUMNS], train["power_kw"]
X_test, y_test = test[COLUMNS], test["power_kw"]
print("train", X_train.shape, " test", X_test.shape)
print(f"mean power_kw in train: {y_train.mean():.2f}")
print()

# Step 2 - baseline
print("== Step 2: the baseline, always the mean ==")
dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
print("dummy predicts:", dummy.predict(X_test[:3]).round(2))
scores("dummy (mean)", y_test, dummy.predict(X_test))
print()

# Step 3 - load only
print("== Step 3: linear regression on load_pct only ==")
load_only = LinearRegression().fit(X_train[["load_pct"]], y_train)
print(f"power_kw = {load_only.coef_[0]:.3f} * load_pct + {load_only.intercept_:.3f}")
scores("linear, load only", y_test, load_only.predict(X_test[["load_pct"]]))
print()

# Step 4 - all six columns
print("== Step 4: linear regression on load, ambient and type ==")
linear = LinearRegression().fit(X_train, y_train)
scores("linear, 6 columns", y_test, linear.predict(X_test))
print()

# Step 5 - coefficients
print("== Step 5: read the coefficients ==")
coefficients = pd.Series(linear.coef_, index=COLUMNS).round(3)
print(coefficients.to_string())
print(f"intercept {linear.intercept_:.3f}")
print()

# Step 6 - error by type and a plot
print("== Step 6: error by machine type, then the plot ==")
result = test[["machine_id", "date", "power_kw"]].copy()
result["predicted"] = linear.predict(X_test).round(1)
result["machine_type"] = test[["type_pump", "type_compressor", "type_conveyor", "type_chiller"]].idxmax(axis=1).str.replace("type_", "")
result["abs_error"] = (result["power_kw"] - result["predicted"]).abs()
print(result.groupby("machine_type")[["power_kw", "abs_error"]].mean().round(2).to_string())
print(result.head(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(result["power_kw"], result["predicted"], s=4, alpha=0.4)
ax.plot([0, 85], [0, 85], color="black", linewidth=1)
ax.set_xlabel("actual power_kw")
ax.set_ylabel("predicted power_kw")
ax.set_title("Test quarter: actual vs predicted")
fig.tight_layout()
fig.savefig("week05/power_actual_vs_predicted.png", dpi=120)
print("saved week05/power_actual_vs_predicted.png")
