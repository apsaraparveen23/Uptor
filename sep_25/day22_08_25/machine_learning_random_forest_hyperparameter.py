import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv('creditcard.csv')
# print(df.head(10))

target_finding = df['Class'].unique()
# print(target_finding)

column_check = df.columns
print(column_check)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators' : [10, 20, 30],
}

param_dist = {
    'n_estimators': [100, 200, 300, 500],        # number of trees
    'max_depth': [None, 10, 20, 30, 50],         # max depth of trees
    'min_samples_split': [2, 5, 10],             # min samples to split a node
    'min_samples_leaf': [1, 2, 4],               # min samples at leaf
    'max_features': ['sqrt', 'log2'],            # features considered per split
    'bootstrap': [True, False],                  # bootstrap sampling
    'class_weight': [None, 'balanced', 'balanced_subsample']  # imbalance handling
}

model = RandomForestClassifier(n_estimators=10) # n_estimators basically to determine how many decision tree to form
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

final_metrics = accuracy_score(y_test, y_pred)
print(final_metrics)

final_metrics1 = confusion_matrix(y_test, y_pred)
print(final_metrics1)

cross_val_scoring = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(cross_val_scoring)





