from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# Load the Iris dataset
iris = load_iris()
data = iris.data
feature_names = iris.feature_names
target = iris.target
target_names = iris.target_names  #  use iris.target_names to get actual names

# Apply Agglomerative Clustering
model = AgglomerativeClustering(n_clusters=3, linkage='single', metric='euclidean')
model.fit(data)
labels = model.labels_
print(labels)

# Map numeric targets to actual species names
actual_target_data = np.array([target_names[i] for i in target])

# Create a DataFrame to compare actual vs. predicted clusters
results_df = pd.DataFrame({
    'actual_target': actual_target_data,
    'cluster_label': labels
})

# Create a cross-tabulation
cross_tab = pd.crosstab(results_df['actual_target'], results_df['cluster_label'])

# Display the cross-tab
print("\nCross-tabulation of actual_target vs. Cluster Labels:")
print(cross_tab)

# Map clusters to dominant categories
print("\nMapping clusters to known categories:")
for cluster_id in cross_tab.columns:
    dominant_category = cross_tab[cluster_id].idxmax()
    count = cross_tab[cluster_id].max()
    print(f"Cluster {cluster_id} likely corresponds to '{dominant_category}' ({count} samples)")
