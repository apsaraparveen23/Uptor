from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train the Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=10, criterion="entropy", max_depth=3, random_state=0)
rf_clf.fit(X_train, y_train)

# Extract and plot one tree from the Random Forest
# plt.figure(figsize=(12, 8))
# plot_tree(rf_clf.estimators_[9], feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
# plt.show()

for x, tree in enumerate(rf_clf.estimators_):
    plt.figure(figsize=(12, 8))
    plot_tree(rf_clf.estimators_[x], feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
    plt.show()

"""random forest is called bagging (bootstrapping sampling)"""
