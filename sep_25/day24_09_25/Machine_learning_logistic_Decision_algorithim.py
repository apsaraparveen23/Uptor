import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# 1. Load dataset

df = pd.read_csv("customer_purchase_behavior.csv")

# 2. Define features and target

X = df.drop(columns=["CustomerID", "PurchaseAmount", "PurchaseDecision"])
y = df["PurchaseDecision"]


# 3. Identify column types

numeric_cols = ["Age", "AnnualIncome", "WebsiteVisits", "TimeOnApp",
                "PreviousPurchases", "LoyaltyPoints"]
categorical_cols = ["Gender", "CampaignResponse", "PreferredDevice", "Region"]


# 4. Preprocessing

imputer_num = SimpleImputer(strategy="median")
X[numeric_cols] = imputer_num.fit_transform(X[numeric_cols])

imputer_cat = SimpleImputer(strategy="most_frequent")
X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])



# Encode categorical
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# 5. Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 6. Logistic Regression


model=LogisticRegression()
model.fit(X_train,y_train)


y_predict=model.predict(X_test)
print(y_predict)

# 7. Decision Tree & Hyperparameter Tuning

tree = DecisionTreeClassifier(random_state=42)

parameters = {
    "max_depth": [3, 5, 10,12,15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}

grid_tree = GridSearchCV(tree, param_grid=parameters, cv=5, scoring="accuracy")
grid_tree.fit(X_train, y_train)

y_pred = grid_tree.predict(X_test)

print("Best Parameters:", grid_tree.best_params_)
print("Best Cross-Validation Accuracy:", grid_tree.best_score_)
print("Accuracy:", accuracy_score(y_test, y_pred))
