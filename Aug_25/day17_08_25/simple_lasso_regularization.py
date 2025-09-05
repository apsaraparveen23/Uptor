import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


my_input ={
    "year":[2000,2001,2002,2003,2004,2005,2006,2007,2008],
     "house_price": [8000,9000,10000,11000,12000,13000,14000,15000,16000]
}#created dictionary

df=pd.DataFrame(my_input)

x=df[["year"]]
y=df["house_price"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model=Lasso(alpha=0.1)
model.fit(x_train,y_train)


y_predict=model.predict(x_test)
print(y_predict)

final_accuracy = r2_score(y_test,y_predict)
print(final_accuracy)