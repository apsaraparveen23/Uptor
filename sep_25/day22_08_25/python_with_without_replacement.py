import numpy as np

data = np.array([1, 2, 3, 4, 5])  # Original dataset

# Sampling WITH replacement (can have duplicates)
with_replacement = np.random.choice(data, size=5, replace=True)

# Sampling WITHOUT replacement (all unique)
without_replacement = np.random.choice(data, size=5, replace=False)

print("With Replacement:", with_replacement)
print("Without Replacement:", without_replacement)
