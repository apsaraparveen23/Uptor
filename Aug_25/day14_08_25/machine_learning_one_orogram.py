from sklearn.linear_model import LinearRegression
import pandas as pd


my_input ={
    "year":[2000,2001,2002,2003,2004,2005,2006,2007,2008],
    "house_price": [8000,9000,10000,11000,12000,13000,14000,15000,16000]
}#created dictionary

df=pd.DataFrame(my_input)

x=df[["year"]]
y=df["house_price"]

model=LinearRegression()
model.fit(x,y)

input_df={"year":[2009,2010]}
input_df=pd.DataFrame(input_df)
my_prediction=model.predict(input_df)
print(my_prediction)