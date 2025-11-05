import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Load dataset
df = pd.read_csv('Mall_Customers.csv')

# Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional: k-distance plot to help select eps
# neigh = NearestNeighbors(n_neighbors=5)
# nbrs = neigh.fit(X_scaled)
# distances, indices = nbrs.kneighbors(X_scaled)
#
# distances = np.sort(distances[:, -1])
# plt.figure(figsize=(8, 4))
# plt.plot(distances)
# plt.title("k-distance Graph (5th Nearest Neighbor Distance)")
# plt.xlabel("Points sorted by distance")
# plt.ylabel("Distance to 5th nearest neighbor")
# plt.grid()
# plt.show()

# Apply DBSCAN
db = DBSCAN(eps=0.5, min_samples=5)  # Adjust eps after checking k-distance plot
labels = db.fit_predict(X_scaled)

# Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', s=60, edgecolor='k')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('DBSCAN Clustering - Mall Customers')
plt.colorbar(label='Cluster Label')
plt.grid(True)
plt.show()

# Analyze clusters
unique, counts = np.unique(labels, return_counts=True)
cluster_info = dict(zip(unique, counts))

print("✅ DBSCAN Cluster Summary:")
for label, count in cluster_info.items():
    if label == -1:
        print(f"Noise points: {count}")
    else:
        print(f"Cluster {label}: {count} points")
