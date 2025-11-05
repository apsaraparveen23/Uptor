"""
**Title:** Clustering with DBSCAN: Density-Based Spatial Clustering of Applications with Noise

**Author:Mohan Sivaraman

**Date:September-20-2025

"""

# DBSCAN project notebook snippet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# 1. Create synthetic dataset
X, y_true = make_blobs(n_samples=500, centers=[[-5,0],[0,0],[5,0]], cluster_std=[0.8,0.5,1.0], random_state=42)
X2, _ = make_circles(n_samples=500, factor=0.5, noise=0.05)

# Choose dataset
X_use = X2  # switch to X or X2

# 2. Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_use)

# 3. (Optional) PCA for visualization if >2D
pca = PCA(n_components=2)
X_vis = pca.fit_transform(X_scaled)

# 4. k-distance plot to choose eps
min_samples = 5
nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)
# distances to k-th neighbor
k_distances = np.sort(distances[:, -1])

plt.figure(figsize=(6,4))
plt.plot(k_distances)
plt.ylabel(f"{min_samples}-NN distance")
plt.xlabel("Points sorted by distance")
plt.title("k-distance graph for eps selection")
plt.grid(True)
plt.show()

# 5. Fit DBSCAN
eps = 0.15  # pick from k-distance elbow
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
labels = db.labels_

# -1 means noise
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)
print("Estimated clusters:", n_clusters)
print("Estimated noise points:", n_noise)

# 6. Evaluation (only if clusters >=2)
if n_clusters >= 2:
    sil = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    print(f"Silhouette: {sil:.3f}, Davies-Bouldin: {dbi:.3f}, Calinski-Harabasz: {ch:.3f}")
else:
    print("Not enough clusters for some metrics.")

# 7. Visualization
plt.figure(figsize=(6,6))
unique_labels = set(labels)
for k in unique_labels:
    if k == -1:
        col = 'k'
        marker = 'x'
        lab = 'noise'
    else:
        col = None  # let matplotlib choose
        marker = 'o'
        lab = f'cluster {k}'
    class_member_mask = (labels == k)
    xy = X_vis[class_member_mask]
    plt.scatter(xy[:,0], xy[:,1], marker=marker, label=lab, alpha=0.7, s=30)

plt.legend()
plt.title(f"DBSCAN clustering (eps={eps}, min_samples={min_samples})")
plt.show()
