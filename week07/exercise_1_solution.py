"""Week 7, Exercise 1 - Cluster the daily readings into operating modes.

Run from the kit root, with the venv active:

    python week07/exercise_1_solution.py

Steps:
  1. load readings.csv and keep four sensor columns;
  2. scale them (K-Means uses distances, so every column must count the same);
  3. K-Means with 3 clusters;
  4. read the cluster means and give each cluster a name (idle / normal / heavy);
  5. silhouette score, and inertia for k = 2 to 6;
  6. save week07/clusters.csv for Exercise 2.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

COLUMNS = ["load_pct", "temperature_c", "vibration_mm_s", "power_kw"]
pd.set_option("display.width", 140)

# Step 1 - the readings, four columns
print("== Step 1: the sensor columns ==")
readings = pd.read_csv("data/clean/readings.csv")
X = readings[COLUMNS]
print(X.describe().round(1))

# Step 2 - scale
print()
print("== Step 2: scale ==")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("mean of each scaled column:", X_scaled.mean(axis=0).round(2))
print("std of each scaled column :", X_scaled.std(axis=0).round(2))

# Step 3 - K-Means, 3 clusters
print()
print("== Step 3: K-Means with 3 clusters ==")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
readings["cluster"] = kmeans.fit_predict(X_scaled)
print(readings["cluster"].value_counts().sort_index())
print(f"inertia: {kmeans.inertia_:.0f}")

# Step 4 - means and names
print()
print("== Step 4: cluster means and names ==")
means = readings.groupby("cluster")[COLUMNS].mean().round(1)
means["rows"] = readings["cluster"].value_counts().sort_index()
print(means)
order = means["load_pct"].sort_values().index
names = {int(order[0]): "idle", int(order[1]): "normal", int(order[2]): "heavy"}
readings["mode"] = readings["cluster"].map(names)
print("names:", names)
readings["weekend"] = (pd.to_datetime(readings["date"]).dt.weekday >= 5).astype(int)
print()
print("share of weekend days per mode:")
print(readings.groupby("mode")["weekend"].mean().round(2))

# Step 5 - silhouette and inertia curve
print()
print("== Step 5: silhouette and the inertia curve ==")
sil = silhouette_score(X_scaled, readings["cluster"], sample_size=5000, random_state=42)
print(f"silhouette (k=3): {sil:.3f}")
inertias = []
for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    score = silhouette_score(X_scaled, model.labels_, sample_size=5000, random_state=42)
    inertias.append({"k": k, "inertia": round(model.inertia_), "silhouette": round(score, 3)})
curve = pd.DataFrame(inertias)
print(curve.to_string(index=False))
plt.figure(figsize=(6, 3.5))
plt.plot(curve["k"], curve["inertia"], marker="o")
plt.xlabel("k (number of clusters)")
plt.ylabel("inertia")
plt.title("K-Means: inertia drops as k grows")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("week07/inertia.png", dpi=120)
print("saved week07/inertia.png")

# Step 6 - save for Exercise 2
print()
print("== Step 6: save ==")
readings[["reading_id", "machine_id", "date", "cluster", "mode", "fault_next_7d"]].to_csv("week07/clusters.csv", index=False)
print("saved week07/clusters.csv", readings.shape)
