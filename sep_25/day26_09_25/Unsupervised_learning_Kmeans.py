import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# 1. Create a sample dataset (Annual Income vs Spending Score)
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'AnnualIncome': [15, 16, 17, 35, 40, 70, 75, 85, 90, 95],
    'SpendingScore': [39, 81, 6, 77, 40, 76, 6, 90, 45, 50]
}
df = pd.DataFrame(data)

print("\nSample Customer Data:\n", df)

# 2. Select features
X = df[['AnnualIncome', 'SpendingScore']]

# 3. Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
df['Cluster'] = kmeans.labels_

print("\nClustered Customer Data:\n", df)

# 4. Visualization
plt.scatter(df['AnnualIncome'], df['SpendingScore'],
            c=df['Cluster'], cmap='viridis', edgecolor='k')

# Plot cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1],
            s=200, c='red', marker='X', label='Centroids')

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.legend()
plt.show()