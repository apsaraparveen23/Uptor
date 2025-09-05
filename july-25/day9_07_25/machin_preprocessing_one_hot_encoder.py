import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv("diamonds.csv")
# print(df)

# one_hot_encoder=OneHotEncoder(sparse_output=False)
# encoder_data=one_hot_encoder.fit_transform(df[['cut']])
# print(encoder_data)

"""Newly generated data frame with columns and thier appropriate names"""

# encoder_df=pd.DataFrame(encoder_data,columns=one_hot_encoder.get_feature_names_out())
# print((encoder_df))
# final_df=pd.concat([df,encoder_df],axis=1)
# print(final_df)
# final_df.drop('cut',axis=1,inplace=True)
# print(final_df)

"""one hot encoding for multiple categorical caolumns"""
one_hot_encoder=OneHotEncoder(sparse_output=False)
encoder_data=one_hot_encoder.fit_transform(df[['cut','color']])
print(encoder_data)
print(df['color'].nunique())
print(df['cut'].nunique())

"""Newly generated data frame with columns and thier appropriate names"""
# encoder_df=pd.DataFrame(encoder_data,columns=one_hot_encoder.get_feature_names_out())
# print((encoder_df))
# final_df=pd.concat([df,encoder_df],axis=1)
# print(final_df)
# final_df.drop('cut',axis=1,inplace=True)
# print(final_df)