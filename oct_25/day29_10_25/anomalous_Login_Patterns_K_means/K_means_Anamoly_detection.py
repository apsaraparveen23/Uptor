#  Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 2: Simulate login data
# Normal logins: short duration, few attempts
normal_logins = np.random.normal(loc=[5, 2], scale=[1, 0.5], size=(100, 2))

# Anomalous logins: long duration, many attempts
anomalous_logins = np.random.normal(loc=[15, 8], scale=[1, 0.5], size=(10, 2))

# Combine data
data = np.vstack((normal_logins, anomalous_logins))
df = pd.DataFrame(data, columns=['login_duration', 'login_attempts'])

#  Step 3: Standardize features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

#  Step 4: Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_data)

#  Step 5: Identify anomaly cluster (smaller group)
anomaly_cluster = df['cluster'].value_counts().idxmin()
df['is_anomaly'] = df['cluster'] == anomaly_cluster

#  Step 6: Visualize clusters
plt.figure(figsize=(8, 6))
colors = ['green' if not anomaly else 'red' for anomaly in df['is_anomaly']]
plt.scatter(df['login_duration'], df['login_attempts'], c=colors, alpha=0.6)
plt.xlabel('Login Duration')
plt.ylabel('Login Attempts')
plt.title('K-Means Clustering: Normal vs Anomalous Logins')
plt.grid(True)
plt.show()

# Step 7: Print flagged anomalies
print("\n🚨 Suspicious login patterns detected:")
print(df[df['is_anomaly']])