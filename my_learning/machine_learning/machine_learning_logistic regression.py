import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# --------------------------
# 1. Create training data
# --------------------------
# X = study hours
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]])
# y = 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])

# --------------------------
# 2. Train logistic regression model
# --------------------------
model = LogisticRegression()
model.fit(X, y)

# --------------------------
# 3. Predict probabilities
# --------------------------
hours = np.linspace(0, 10, 100).reshape(-1, 1)  # from 0 to 10 hours
probabilities = model.predict_proba(hours)[:, 1]  # probability of passing


# --------------------------
# 4. Plot results
# --------------------------
plt.scatter(X, y, color="red", label="Actual data (0=Fail, 1=Pass)")
plt.plot(hours, probabilities, color="blue", linewidth=2, label="Logistic curve (Pass probability)")

plt.axhline(0.5, color="green", linestyle="--", label="Decision boundary (0.5)")
plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression: Study Hours vs Pass/Fail")
plt.legend() # identify what each color,line or marker represents
plt.show()



