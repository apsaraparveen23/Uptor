import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

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

param_grid = {
    'n_estimators' : [10, 20, 30],
}



model = RandomForestClassifier(n_estimators=10) # n_estimators basically to determine how many decision tree to form
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('-----------------')
print(y_pred)

final_metrics = accuracy_score(y_test, y_pred)
print('******************')
print(final_metrics)

final_metrics1 = confusion_matrix(y_test, y_pred)
print('******************')
print(final_metrics1)

cross_val_scoring = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print('******************')
print(cross_val_scoring)





