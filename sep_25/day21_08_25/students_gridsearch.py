import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("students_study_dataset.csv")

x= df[["StudyHours", "SleepHours"]]
y = df["Passed"]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

model = DecisionTreeClassifier(random_state=42)
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5, 6, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 7. Perform Grid Search with Cross Validation
grid_search = GridSearchCV(estimator=model,
                           param_grid=parameters,
                           cv=2,
                           scoring='accuracy',
                           n_jobs=-1)

# 8. Fit Grid Search
grid_search.fit(x_train,y_train)

# 9. Output Best Parameters and Accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# 10. Optional: Use the best model for predictions
best_model = grid_search.best_estimator_

# Predict on the same data or use train_test_split for validation
y_pred = best_model.predict(x_test)

# Evaluate on training data (you can also split the data for better validation)
from sklearn.metrics import accuracy_score
print("Accuracy on training set:", accuracy_score(y_test, y_pred))