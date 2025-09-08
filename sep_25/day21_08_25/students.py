import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

df = pd.read_csv("students_study_dataset.csv")

x = df[["StudyHours", "SleepHours"]]
y = df["Passed"]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

model = DecisionTreeClassifier(random_state=42, max_depth=2)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Predictions:", y_pred)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
plt.figure(figsize=(10,6))
plot_tree(model,
          feature_names=["StudyHours", "SleepHours"],
          class_names=["Fail", "Pass"],
          filled=True,
          rounded=True)
plt.title("Decision Tree Visualization")
plt.show()
