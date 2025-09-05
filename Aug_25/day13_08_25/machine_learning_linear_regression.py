import pandas as pd
from sklearn.linear_model import LinearRegression
from utilities.file_reading import data_to_data_frame


"""below input contains year and house price for the data frame"""
my_input ={
    "year":[2000,2001,2002,2003,2004,2005,2006,2007,2008],
    "house_price": [8000,9000,10000,11000,12000,13000,14000,15000,16000]
}#created dictionary

# df=data_to_data_frame(my_input)#converting data in to data frame
# print(df)
# x=df[["year"]]
# y=df["house_price"]
# model=LinearRegression()#intailizing linear regression
# model.fit(x,y)#training
#
# """converting new years into a DataFrame and feeding them to the trained model to get predicted prices."""
#
# my_input={"year":[2009,2010]}
# input_df=data_to_data_frame((my_input))
# my_prediction=model.predict(input_df)
# print(my_prediction)

df=data_to_data_frame(my_input)
print(df)
x=df[["year"]]
y=df["house_price"]

def model_training(model_name):
    try:
        model=model_name
        model.fit(x,y)
        return model
    except Exception as ex:
        return ex
