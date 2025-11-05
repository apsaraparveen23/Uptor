# train_decision_tree.py
import numpy as np
import pickle
from sklearn.tree import DecisionTreeRegressor

# Sample training data (e.g., year vs value)
X = np.array([[2000], [2005], [2010], [2015], [2020]])
y = np.array([100, 150, 200, 250, 300])

# Train the model
model = DecisionTreeRegressor()
model.fit(X, y)

# Save the model
with open("uptor_203_tree_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Decision Tree model saved as 'uptor_203_tree_model.pkl'")