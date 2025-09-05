import pandas as pd
from sklearn.linear_model import LinearRegression

df=pd.read_csv("Car Sell Dataset1.csv")
print(df)
x=df[['Year']]
y=df['Price']
model=LinearRegression()
model.fit(x,y)

new_data=pd.read_csv("Car Sell Dataset1.csv")
