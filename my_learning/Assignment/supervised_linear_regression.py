import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv("diamonds.csv")
print("Dataset Loaded Successfully")
print(df.head())

# Step 2: Handle missing values (if any)
df = df.dropna()

# Step 3: Encode categorical column 'cut'
le = LabelEncoder()
df['cut_encoded'] = le.fit_transform(df['cut'])

# Step 4: Select features (independent variables) and target (dependent variable)
X = df[['carat', 'cut_encoded']]
y = df['price']

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\n  Model Evaluation:")
print("Mean Squared Error:", round(mse, 2))
print("R² Score:", round(r2, 4))

# Step 9: Predict price for a new diamond
# Example: diamond with carat = 1.2 and cut = 'Premium'
cut_label = le.transform(['Premium'])[0]
new_data = pd.DataFrame({'carat': [1.2], 'cut_encoded': [cut_label]})
predicted_price = model.predict(new_data)
print("\n Predicted Price for 1.2 carat Premium cut diamond:", round(predicted_price[0], 2))

# Step 10: Show actual vs predicted comparison (first 10)
results = pd.DataFrame({'Actual': y_test.head(10), 'Predicted': y_pred[:10]})
print("\n Actual vs Predicted Prices:")
print(results)
