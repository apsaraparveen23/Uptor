# 1. Import libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd

# 2. Load the iris dataset
iris = load_iris()
X = iris.data   # features (sepal length, sepal width, petal length, petal width)

# 3. Create a DataFrame (optional, just for easy view)
# df = pd.DataFrame(X, columns=iris.feature_names)
# print(df.head())

# 4. Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # we know Iris has 3 species
kmeans.fit(X)

cluster_labels =kmeans.labels_

# # 5. Get cluster labels for each row
# df['Cluster'] = kmeans.labels_
# print(df.head())

# 6. Compare clusters with actual species (optional)
# print("Cluster centers:\n", kmeans.cluster_centers_)

plt.figure(figsize=(8,6))
scatter=plt.scatter(X[:,0],X[:,1],c=cluster_labels, cmap='viridis')
plt.title("K-means clustering on Iris Dataset")
plt.xlabel("sepal length(cm)")
plt.ylabel("sepal width(cm)")
legend1=plt.legend(*scatter.legend_elements(),title="clusters")
plt.show()
print("cluster labels assigned by K means:")
print(cluster_labels)
print("\n True label of dataset")
