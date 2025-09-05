import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder


df= pd.read_csv("diamonds.csv")
print(df)

columns = df.columns
print(columns)

finding_null=df.isnull().sum()
print(finding_null)

df.dropna(inplace=True)

le = LabelEncoder()
df['cut'] = le.fit_transform(df['cut'])

le = LabelEncoder()
df['clarity'] = le.fit_transform(df['clarity'])

le = LabelEncoder()
df['color'] = le.fit_transform(df['color'])


df.drop('Unnamed: 0',axis=1,inplace=True)
print(df)
x= df.drop("price",axis=1)
y= df["price"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)

y_predict = model.predict(x_test)
print(y_predict)

final_accuracy = r2_score(y_test,y_predict)
print(final_accuracy)