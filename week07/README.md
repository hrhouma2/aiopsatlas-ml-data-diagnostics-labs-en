# Week 7 — Clustering and neural networks: solution scripts

Run every script from the kit root, with the venv active. No Ollama needed this week.

| Script | What it does | Writes |
| --- | --- | --- |
| `exercise_1_solution.py` | Exercise 1 — Cluster the 14,600 daily readings into operating modes: scale `load_pct`, `temperature_c`, `vibration_mm_s`, `power_kw`, `KMeans(n_clusters=3, random_state=42)`, name the clusters from their mean load (idle 5,380 rows, normal 6,211, heavy 3,009), silhouette 0.256, inertia for k = 2 to 6. | `clusters.csv`, `inertia.png` |
| `exercise_2_solution.py` | Exercise 2 — PCA to 2D on the same scaled columns (PC1 62.9 %, PC2 17.0 %), two scatter plots coloured by mode and by `fault_next_7d`, fault rate per mode (idle 4.5 %, normal 7.6 %, heavy 12.3 %). Rebuilds the clusters if `clusters.csv` is missing. | `pca_clusters.png` |
| `exercise_3_solution.py` | Exercise 3 — `MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True)` vs logistic regression on the Week 4 features, temporal split: logistic 0.961, MLP 0.945 in 26 epochs, MLP without early stopping 0.923 in 300 epochs. Rebuilds `data/features.csv` if missing. | `mlp_loss.png` |

Run 1 before 2. Exercise 3 is independent.
