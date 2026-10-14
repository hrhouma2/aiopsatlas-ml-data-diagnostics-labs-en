# Week 5 — Regression, classification and evaluation: solution scripts

Run every script from the kit root, with the venv active. All three read `data/features.csv`, built in Week 4 Exercise 1 (or with `python common/features.py`).

| Script | What it does |
| --- | --- |
| `exercise_1_solution.py` | Exercise 1 — Predict `power_kw` from `load_pct`, `ambient_c` and the four `type_*` columns. `DummyRegressor` (R2 0.000) against `LinearRegression` (R2 0.933, MAE 3.20). Reads the coefficients. Saves `power_actual_vs_predicted.png`. |
| `exercise_2_solution.py` | Exercise 2 — Predict `fault_next_7d` with the 15 Week 4 features, temporal split. `DummyClassifier` accuracy 0.936 against a logistic regression: AUC 0.961, confusion matrix `[[3432, 11], [83, 154]]`, precision 0.933, recall 0.650. |
| `exercise_3_solution.py` | Exercise 3 — Move the threshold from 0.5 down to 0.1, `class_weight="balanced"`, a cost per threshold (missed fault 1,200 CAD, false alarm 150 CAD). Saves `threshold_curves.png` and `precision_recall.png`. |

The PNG files are written next to the scripts when you run them.
