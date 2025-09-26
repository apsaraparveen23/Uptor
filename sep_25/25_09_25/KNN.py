# 1. Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 2. Create original dataset
X = np.array([
    # Class 0 points(Normal patients)
    [2, 5], [3, 6], [4, 5], [5, 7], [6, 6], [5, 5], [4, 6], [6, 7], [7, 6], [6, 5],
    # Class 1 points(Risk patients)
    [7, 2], [8, 3], [9, 2], [10, 4], [9, 5], [8, 4], [10, 3], [11, 5], [7, 4], [8, 5]
])
y = np.array([0]*10 + [1]*10)

# 3. Add slight noise to simulate real-world imperfections
np.random.seed(42)
X_noisy = X + np.random.normal(0, 0.4, X.shape)

# 4. Flip a couple of labels to simulate mislabeled data
y_noisy = y.copy()
y_noisy[5] = 1  # Flip one class 0 to class 1
y_noisy[15] = 0 # Flip one class 1 to class 0

# 5. Split into training and testing sets (50/50 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_noisy, y_noisy, test_size=0.5, random_state=42
)

# 6. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train KNN with smaller k (e.g., k=3)
k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

# 8. Predictions and evaluation
y_pred = knn_classifier.predict(X_test)
train_acc = accuracy_score(y_train, knn_classifier.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)

print(f"Train Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")
print(f"Predictions: {y_pred}")
print(f"Actual Labels: {y_test}")

# 9. Accuracy vs. k plot (limit k to ≤ training samples)
k_values = range(1, len(X_train) + 1)
train_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

plt.plot(k_values, train_scores, label='Train Accuracy')
plt.plot(k_values, test_scores, label='Test Accuracy')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs. k (with noise)')
plt.legend()
plt.grid(True)
plt.show()

# 10. Cross-validation
cv_scores = cross_val_score(knn_classifier, scaler.transform(X_noisy), y_noisy, cv=5)
print(f"Cross-validated Accuracy (k={k}): {cv_scores.mean():.2f}")