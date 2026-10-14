"""Week 7, Exercise 2 - PCA to 2D, coloured by cluster and by fault.

Run from the kit root, with the venv active:

    python week07/exercise_2_solution.py

Steps:
  1. load readings.csv and the clusters of Exercise 1 (rebuilt here if week07/clusters.csv is missing);
  2. scale the same four columns;
  3. PCA with 2 components: how much of the variance do they keep?
  4. read the loadings: what does each axis mean?
  5. two scatter plots side by side, saved to week07/pca_clusters.png;
  6. fault rate per operating mode.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

COLUMNS = ["load_pct", "temperature_c", "vibration_mm_s", "power_kw"]
pd.set_option("display.width", 140)


def load_clusters(readings, X_scaled, path="week07/clusters.csv"):
    """Exercise 1 saved the clusters. Rebuild them with the same K-Means if the file is missing."""
    if os.path.exists(path):
        saved = pd.read_csv(path)
        return saved["cluster"].to_numpy(), saved["mode"].to_numpy()
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
    labels = kmeans.labels_
    order = pd.Series(readings["load_pct"].to_numpy()).groupby(labels).mean().sort_values().index
    names = {order[0]: "idle", order[1]: "normal", order[2]: "heavy"}
    return labels, pd.Series(labels).map(names).to_numpy()


# Step 1 - readings and clusters
print("== Step 1: readings and clusters ==")
readings = pd.read_csv("data/clean/readings.csv")
X_scaled = StandardScaler().fit_transform(readings[COLUMNS])
readings["cluster"], readings["mode"] = load_clusters(readings, X_scaled)
print(readings["mode"].value_counts())

# Step 2 and 3 - PCA
print()
print("== Step 3: PCA with 2 components ==")
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
readings["pc1"], readings["pc2"] = coords[:, 0], coords[:, 1]
ratio = pca.explained_variance_ratio_
print(f"variance kept: PC1 {ratio[0]:.1%}, PC2 {ratio[1]:.1%}, total {ratio.sum():.1%}")

# Step 4 - loadings
print()
print("== Step 4: loadings ==")
loadings = pd.DataFrame(pca.components_, columns=COLUMNS, index=["PC1", "PC2"]).round(2)
print(loadings)

# Step 5 - two pictures
print()
print("== Step 5: plot ==")
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
colours = {"idle": "tab:blue", "normal": "tab:green", "heavy": "tab:red"}
for mode, part in readings.groupby("mode"):
    axes[0].scatter(part["pc1"], part["pc2"], s=4, alpha=0.4, color=colours[mode], label=f"{mode} ({len(part)})")
axes[0].set_title("Coloured by operating mode (K-Means)")
axes[0].legend(markerscale=4)
quiet = readings[readings["fault_next_7d"] == 0]
faults = readings[readings["fault_next_7d"] == 1]
axes[1].scatter(quiet["pc1"], quiet["pc2"], s=4, alpha=0.2, color="lightgrey", label=f"no fault ({len(quiet)})")
axes[1].scatter(faults["pc1"], faults["pc2"], s=6, alpha=0.7, color="black", label=f"fault_next_7d = 1 ({len(faults)})")
axes[1].set_title("Coloured by fault_next_7d")
axes[1].legend(markerscale=4)
for ax in axes:
    ax.set_xlabel(f"PC1 ({ratio[0]:.0%} of the variance)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel(f"PC2 ({ratio[1]:.0%} of the variance)")
plt.tight_layout()
plt.savefig("week07/pca_clusters.png", dpi=120)
print("saved week07/pca_clusters.png")

# Step 6 - do the modes know about faults?
print()
print("== Step 6: fault rate per mode ==")
rate = readings.groupby("mode")["fault_next_7d"].agg(["mean", "sum", "count"]).round(3)
rate.columns = ["fault_rate", "faults", "rows"]
print(rate.sort_values("fault_rate"))
print()
print("mean position of the two groups:")
print(readings.groupby("fault_next_7d")[["pc1", "pc2"]].mean().round(2))
