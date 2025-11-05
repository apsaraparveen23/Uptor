import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Load dataset (Mall Customers)
data = pd.read_csv("Mall_Customers.csv")

# Use just two features for easy plotting
X = data[["Annual Income (k$)", "Spending Score (1-100)"]].values

# Scale the data
X_scaled = StandardScaler().fit_transform(X)

# Apply DBSCAN
db = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
labels = db.labels_

# Plot results
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=labels, cmap="plasma", s=50, alpha=0.7)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("DBSCAN Clustering on Mall Customers")
plt.colorbar(label="Cluster Label")
plt.show()
