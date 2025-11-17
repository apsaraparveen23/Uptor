import numpy as np
import pandas as pd
from statistics import mean, median, mode, stdev, variance

# Step 1: Create sample data
data = [45, 50, 55, 60, 65, 70, 75, 80, 85, 90]

# Step 2: Calculate basic statistics
print(" Basic Statistics")
print("Data:", data)
print("Mean:", mean(data))
print("Median:", median(data))
print("Mode:", mode(data))
print("Standard Deviation:", stdev(data))
print("Variance:", variance(data))

# Step 3: Using Pandas for more insights
df = pd.DataFrame({
    'Height(cm)': [150, 155, 160, 165, 170, 175, 180],
    'Weight(kg)': [50, 53, 57, 63, 67, 72, 75]
})

print("\n Dataset:\n", df)

# Step 4: Statistical summary using Pandas
print("\nDescriptive Statistics:")
print(df.describe())

# Step 5: Correlation between Height and Weight
print("\nCorrelation between Height and Weight:")
print(df.corr())
