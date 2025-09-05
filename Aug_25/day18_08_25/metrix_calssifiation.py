import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


my_input ={
    "house_price": [8000,900,1000,11000,1200,1300,14000,15000,1600],
    "house_purchase":[0,1,1,0,1,1,0,0,1]
}#created dictionary
#values given indicates higher value-0(not purchasing),lower value-1(purchasing)

df=pd.DataFrame(my_input)

x=df[["house_price"]]
y=df["house_purchase"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)


y_predict=model.predict(x_test)
print(y_predict)#gives binary values

final_accuracy = accuracy_score(y_test,y_predict)
print(final_accuracy)
print("------")
confusion_matrix_check = confusion_matrix(y_test,y_predict)
print(confusion_matrix_check)
print("------")
classification_check =classification_report(y_test,y_predict)
print(classification_check)
