# Week 6 — Model selection and validation: solution scripts

Run every script from the kit root, with the venv active. No Ollama needed this week.

The scripts use the Week 4 feature table. They read `data/features.csv` when it exists and rebuild it from `data/clean/` otherwise, with the same `make_features()` helper at the top of each file.

| Script | What it does | Writes |
| --- | --- | --- |
| `exercise_1_solution.py` | Exercise 1 — Tree depth sweep: fit on Jan-Jun, validate on Jul-Sep, `max_depth` from 1 to 20, train vs validation AUC curve. Best depth 4 (AUC 0.870); depth 20 memorises (1.000 vs 0.726). | `depth_sweep.png` |
| `exercise_2_solution.py` | Exercise 2 — Grid search on a random forest with `TimeSeriesSplit(n_splits=3)`: 12 settings, 36 fits, about half a minute. Best: no depth limit, `min_samples_leaf=20`, 300 trees, mean AUC 0.928. | `best_params.json` |
| `exercise_3_solution.py` | Exercise 3 — Final test on Oct-Dec, used once: tuned forest AUC 0.936, logistic regression 0.961; feature importance; errors by machine type and by month. | `results.md`, `feature_importance.png` |

Run them in order: Exercise 3 reads `best_params.json` from Exercise 2 (and falls back to the same values if the file is missing).
