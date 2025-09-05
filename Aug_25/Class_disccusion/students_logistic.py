import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv("StudentsPerformance.csv")

my_input ={
    "math score": [80,70,20,10,85,15,75,25],
    "pass_fail":[1,1,0,0,1,0,1,0]
}

df=pd.DataFrame(my_input)

x = df[["math score"]]
y = df["pass_fail"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(x_train,y_train)


y_predict=model.predict(x_test)
print(y_predict)
print("----------------------")
final_accuracy = accuracy_score(y_test,y_predict)
print(final_accuracy)
