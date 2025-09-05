import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv("diamonds.csv")
# print(df)

""""""
encoder_df=pd.get_dummies(df['cut'])
print(encoder_df)
final_df=pd.concat([df,encoder_df],axis=1)
print(final_df)

