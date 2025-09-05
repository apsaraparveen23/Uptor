import pandas as pd
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("diamonds.csv")
print(df)

standard_scaler_object=StandardScaler()
df['price']=standard_scaler_object.fit_transform(df[['price']])
print(df)