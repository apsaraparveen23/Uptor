import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df=pd.read_csv("house_price_bd.csv")
print(df)
df.dropna(inplace=True)
column_data_types=df.dtypes
print(column_data_types)


df['Price_in_taka']=df['Price_in_taka'].replace({"৳":"",",":""},regex=True).astype(int)

x=df.drop(["Price_in_taka","Title","City","Location","Floor_no","Occupancy_status"],axis=1)
y=df['Price_in_taka']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)

y_predict=model.predict(x_test)

print(y_predict)
accuracy_check=r2_score(y_test,y_predict)
print(accuracy_check)