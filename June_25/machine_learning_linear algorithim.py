import pandas as pd
from sklearn.linear_model import LinearRegression

# Step 1: Create dataset

data={
    "size_sqft":[1000,1500,2000,2500,3000,3500],
    "price_lakhs":[50,75,100,125,150,175]
}
df=pd.DataFrame(data)

# Step 2: Input (x) and Output (y)

x=df[['size_sqft']]
y=df['price_lakhs']

# Step 3: Train model

model=LinearRegression()
model.fit(x,y)

# Step 4: Predict for a new

new_house_sqft=[[4000]]
new_house = pd.DataFrame({"size_sqft": [4000]})
predicted_price=model.predict(new_house)
print("Predicted price for new house:",predicted_price[0],"lakhs")