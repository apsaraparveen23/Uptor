import pandas as pd

from sklearn.preprocessing import StandardScaler,MinMaxScaler

df=pd.read_csv("diamonds.csv")

"Standard scaling"
standard_scaler = StandardScaler()
df[['price']]= standard_scaler.fit_transform(df[['price']])
print(df)

#After standard scaling, the mean becomes 0 and the standard deviation becomes 1.

standard_mean=df[['price']].mean()
print(standard_mean)
standard_std=df[['price']].std()
print(standard_std)

"MinMaxScaling"

standard_min_max = MinMaxScaler()
df[['price']] = standard_min_max.fit_transform(df[['price']])
print(df)

#After MinMax scaling, the 'price' values are compressed into the range [0, 1]
standard_mean=df[['price']].mean()
print(standard_mean)
standard_std=df[['price']].std()
print(standard_std)