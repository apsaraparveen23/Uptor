import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset
df = pd.read_csv("diamonds.csv")
print("Dataset Loaded Successfully")

# Step 2: Handle missing values
df = df.dropna()

# Step 3: Encode categorical columns
le = LabelEncoder()
df['cut_encoded'] = le.fit_transform(df['cut'])

# Step 4: Select features and target
X = df[['carat', 'cut_encoded']]
y = df['price']

# Step 5: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Define the model
rf = RandomForestRegressor(random_state=42)

# Step 7: Define the parameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 150],        # Number of trees
    'max_depth': [None, 5, 10, 20],        # Depth of trees
    'min_samples_split': [2, 5, 10],       # Min samples to split a node
    'min_samples_leaf': [1, 2, 4]          # Min samples at a leaf node
}

# Step 8: Apply GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='r2')

grid_search.fit(X_train, y_train)

# Step 9: Get best parameters
print("\n Best Parameters Found:", grid_search.best_params_)

# Step 10: Train final model with best parameters
best_model = grid_search.best_estimator_

# Step 11: Evaluate the tuned model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation After Tuning:")
print("Mean Squared Error:", round(mse, 2))
print("R² Score:", round(r2, 4))

# Step 12: Predict a new diamond’s price
cut_label = le.transform(['Premium'])[0]
new_data = pd.DataFrame({'carat': [1.2], 'cut_encoded': [cut_label]})
predicted_price = best_model.predict(new_data)
print("\n  Predicted Price for 1.2 carat Premium cut diamond:", round(predicted_price[0], 2))
