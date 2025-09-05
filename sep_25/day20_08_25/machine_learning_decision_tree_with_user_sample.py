import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Sample data
data = {
    'Weather': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Overcast', 'Sunny'],
    'Pitch': ['Dry', 'Dry', 'Damp', 'Damp', 'Dry', 'Dry', 'Damp'],
    'Humidity': ['High', 'Normal', 'High', 'High', 'Normal', 'Normal', 'Normal'],
    'Play': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Convert categorical variables to numeric
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for column in ['Weather', 'Pitch', 'Humidity', 'Play']:
    df[column] = le.fit_transform(df[column])

# Features and Target
X = df[['Weather', 'Pitch', 'Humidity']]
y = df['Play']

# Train Decision Tree
model = DecisionTreeClassifier(criterion='entropy')  # You can also try 'gini'
model.fit(X, y)

# Visualize the tree
plt.figure(figsize=(10, 6))
tree.plot_tree(model, feature_names=['Weather', 'Pitch', 'Humidity'], class_names=['No', 'Yes'], filled=True)
plt.show()
