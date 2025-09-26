import numpy as np
from collections import Counter

# --- 1. Define the Euclidean distance function ---
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

# --- 2. Implement the KNN algorithm ---
def knn_predict(X_train, y_train, new_point, k=3):
    distances = []
    for i, train_point in enumerate(X_train):
        dist = euclidean_distance(train_point, new_point)
        distances.append((dist, y_train[i]))

    # Sort by distance and get the k nearest neighbors
    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = distances[:k]

    # Get the labels of the k nearest neighbors
    neighbor_labels = [label for _, label in k_nearest_neighbors]

    # Predict the class based on the majority vote
    most_common = Counter(neighbor_labels).most_common(1)
    return most_common[0][0]

# --- 3. Create a synthetic dataset (Customer Feedback Example) ---
# X_train: Features (Sentiment Score, Response Time)
# y_train: Labels (Positive, Negative)
X_train = [
    [0.8, 10],  # Positive feedback, fast response
    [0.9, 12],  # Positive feedback, fast response
    [0.7, 15],  # Positive feedback, medium response
    [0.2, 60],  # Negative feedback, slow response
    [0.1, 70],  # Negative feedback, slow response
    [0.3, 55],  # Negative feedback, medium-slow response
    [0.6, 20],  # Positive feedback, medium response
    [0.4, 45]   # Negative feedback, medium-slow response
]
y_train = ['Positive', 'Positive', 'Positive', 'Negative', 'Negative', 'Negative', 'Positive', 'Negative']

# --- 4. Define a new data point for prediction ---
# A new customer feedback with a sentiment score of 0.5 and response time of 30
new_customer_feedback = [0.5, 30]

# --- 5. Make a prediction ---
predicted_class = knn_predict(X_train, y_train, new_customer_feedback, k=3)

print(f"The new customer feedback with sentiment score {new_customer_feedback[0]} and response time {new_customer_feedback[1]} is classified as: {predicted_class}")

# Another example
new_customer_feedback_2 = [0.15, 80]
predicted_class_2 = knn_predict(X_train, y_train, new_customer_feedback_2, k=3)
print(f"The new customer feedback with sentiment score {new_customer_feedback_2[0]} and response time {new_customer_feedback_2[1]} is classified as: {predicted_class_2}")