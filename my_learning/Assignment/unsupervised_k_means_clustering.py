
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Create some random data (2 features)
np.random.seed(42)
x = np.random.rand(100, 2) * 10   # 100 data points, 2 features

# Convert to DataFrame
df = pd.DataFrame(x, columns=['Feature1', 'Feature2'])
print("Dataset sample:")
print(df.head())

# Step 2: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df)

# Step 3: Visualize the clusters
plt.figure(figsize=(7,5))
plt.scatter(df['Feature1'], df['Feature2'], c=df['Cluster'], cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            color='red', marker='X', s=200, label='Cluster Centers')
plt.title('K-Means Clustering Example')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.show()

# Step 4: Display the cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)
