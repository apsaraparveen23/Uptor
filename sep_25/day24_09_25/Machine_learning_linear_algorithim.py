import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("customer_purchase_behavior.csv")
print("Columns in dataset:", df.columns.tolist())

# 2. Define numeric and categorical columns
numeric_cols = ["Age", "AnnualIncome", "WebsiteVisits", "TimeOnApp",
                "PreviousPurchases", "LoyaltyPoints"]
categorical_cols = ["Gender", "CampaignResponse", "PreferredDevice", "Region"]

# 3. Handle missing values
# Numeric → fill with median
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical → fill with most frequent
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 4. Encode categorical features
for col in categorical_cols:
    df[col] = df[col].astype(str)  # ensure all values are strings
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 5. Scale numeric features
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 6. Define features (X) and target (y)
X = df.drop(columns=["CustomerID", "PurchaseAmount", "PurchaseDecision"])
y = df["PurchaseAmount"]

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 8. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 9. Predict and evaluate
y_pred = model.predict(X_test)
print(y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(r2)