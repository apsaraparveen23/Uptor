import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample data
data = {
"year": [2000, 2001, 2002, 2003, 2004],
"price": [1000, 2000, 3000, 4000, 5000] # y = 2x (perfect linear relation)
}

df = pd.DataFrame(data)

X = df[["year"]] # features
y = df["price"] # target

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model to pickle file
with open("upor_203_linear_model.pkl", "wb") as obj:
    pickle.dump( model, obj)
print("Model trained and saved as model.pkl")