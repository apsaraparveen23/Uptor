import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("diamonds.csv")
print(df)

standard_scaler_object=MinMaxScaler()
df['price']=standard_scaler_object.fit_transform(df[['price']])
print(df)

"plot"
df['price'].plot(kind='hist', bins=30, color='yellow', edgecolor='black')#histograph
plt.xlabel("value")
plt.ylabel("frequency")
print(df.ndim)
plt.show()