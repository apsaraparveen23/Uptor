import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("creditcard.csv")

# Drop index-like column
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

# Target column
y = df['is_fraud']
X = df.drop(columns=['is_fraud'])

# Encode categorical (non-numeric) columns
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Predictions:", y_pred[:20])  # show first 20 predictions
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Cross-validation
cross_val_scoring = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-validation scores:", cross_val_scoring)
print("Average CV accuracy:", cross_val_scoring.mean())