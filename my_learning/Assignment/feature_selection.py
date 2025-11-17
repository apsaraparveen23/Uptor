import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

print(" Dataset sample:")
print(X.head())

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a RandomForest model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 4: Get feature importances
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\n Feature Importance:")
print(importance)

# Step 5: Visualize the feature importance
plt.figure(figsize=(8,5))
importance.plot(kind='bar', color='skyblue')
plt.title("Feature Importance from Random Forest (California Housing)")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.show()
