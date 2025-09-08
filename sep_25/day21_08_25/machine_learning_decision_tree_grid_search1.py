import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

# 1. Load Data
column_names = ['buying_price', 'maintenance_cost', 'doors_numbers',
                'person_number', 'luggage_space', 'safety', 'class']
df = pd.read_csv("car_evaluation.csv", names=column_names, header=None)

# 2. Encode Target Variable
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])

# 3. Split Features and Target
X = df.drop("class", axis=1)
y = df['class']

# 4. Ordinal Encode Categorical Features
encoder = ce.OrdinalEncoder(cols=['buying_price', 'maintenance_cost', 'doors_numbers',
                                  'person_number', 'luggage_space', 'safety'])
X_encoded = encoder.fit_transform(X)

# 5. Define Model
dt_model = DecisionTreeClassifier(random_state=42)

# 6. Define Parameter Grid for GridSearch
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5, 6, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 7. Perform Grid Search with Cross Validation
grid_search = GridSearchCV(estimator=dt_model,
                           param_grid=parameters,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1)

# 8. Fit Grid Search
grid_search.fit(X_encoded, y)

# 9. Output Best Parameters and Accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# 10. Optional: Use the best model for predictions
best_model = grid_search.best_estimator_

# Predict on the same data or use train_test_split for validation
y_pred = best_model.predict(X_encoded)

# Evaluate on training data (you can also split the data for better validation)
from sklearn.metrics import accuracy_score
print("Accuracy on full training set:", accuracy_score(y, y_pred))
