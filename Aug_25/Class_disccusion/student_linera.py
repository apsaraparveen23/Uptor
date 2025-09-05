import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


df=pd.read_csv("StudentsPerformance.csv")
# print(df)
# print(df.columns)

x = df[["reading score", "writing score"]]
y = df["math score"]


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


model = LinearRegression()
model.fit(x_train,y_train)

y_predict = model.predict(x_test)
print(y_predict)
print("----------------------")
final_accuracy = r2_score(y_test,y_predict)
print(final_accuracy)
