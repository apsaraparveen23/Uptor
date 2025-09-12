import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score,RandomizedSearchCV

df = pd.read_csv('creditcard_for_random_Forest.csv')
# print(df.head(10))

target_finding = df['Class'].unique()
# print(target_finding)

column_check = df.columns
print(column_check)

# print("NaN in target column:", df['Class'].isna().sum())
# print("Unique values in Class:", df['Class'].unique())

df = df.dropna(subset=['Class'])
df['Class'] = df['Class'].fillna(0)


X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# param_grid = {
#     'n_estimators' : [10, 20, 30],
# }

param_dist = {
    'n_estimators': [100, 200, 300, 500],        # number of trees
    'max_depth': [None, 10, 20, 30, 50],         # max depth of trees
    'min_samples_split': [2, 5, 10],             # min samples to split a node
    'min_samples_leaf': [1, 2, 4],               # min samples at leaf
    'max_features': ['sqrt', 'log2'],            # features considered per split
    'bootstrap': [True, False],                  # bootstrap sampling
    'class_weight': [None, 'balanced', 'balanced_subsample']  # imbalance handling
}

# RandomForest model
model = RandomForestClassifier(random_state=42)

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,            # number of random combinations to try
    cv=5,                 # 5-fold cross-validation
    scoring='f1',         # better for imbalanced data than accuracy
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit model
random_search.fit(X_train, y_train)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

final_metrics = accuracy_score(y_test, y_pred)
print(final_metrics)

final_metrics1 = confusion_matrix(y_test, y_pred)
print(final_metrics1)

cross_val_scoring = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(cross_val_scoring)





