from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

####### Load the IRIS DATASET

iris = load_iris()
X = iris.data
y= iris.target

##### Define the Hyperaparameters to be searched

param_grid = {'max_depth':[2, 4, 6, 8, 10],
              'min_samples_split': [2, 4, 6, 8, 10],
              'min_samples_leaf': [1, 2, 3, 4, 5],
              'criterion': ["gini", "entropy"]}


######## Train the model
X_train, X_test,y_train, y_test =train_test_split(X,y,test_size=0.3,random_state=0)
rf_clf = RandomForestClassifier(n_estimators=10, criterion="entropy", max_depth=3, random_state=0)  ### n_estimators is a number of trees
#rf_clf.fit(X_train,y_train)

###### Create a Grid Search Object
grid_search = GridSearchCV(rf_clf, param_grid, cv=5)  #### CV=5, cross validation taking 5 different sample of data
grid_search.fit(X_train, y_train)

Best_Estimator = grid_search.best_estimator_
##### Print the best hyperparamters
print("Best Hyperparameters :",grid_search.best_params_)


###### Extract and plot one tree from the random forest using For loop
for x, tree in enumerate(Best_Estimator.estimators_):
    plt.figure(figsize=(12, 8))
    plot_tree(tree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
    plt.show()