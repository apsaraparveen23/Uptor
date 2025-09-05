import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

df=pd.read_csv("diamonds.csv")

# "Converting categorical data in to numbers "
#
# ordinal_encoder=OrdinalEncoder()
# df[['cut','color']]=ordinal_encoder.fit_transform(df[['cut','color']])
# print(df)
#
# "Ordinal decoding "
#
# ordinal_encoder=OrdinalEncoder()
# df[['cut','color']]=ordinal_encoder.fit_transform(df[['cut','color']])
#
# ordinal_decoder=OrdinalEncoder()
# df[['cut','color']] = ordinal_encoder.inverse_transform(df[['cut','color']])
# print(df)

