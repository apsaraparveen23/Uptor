import pandas as pd

from sklearn.preprocessing import OneHotEncoder
df=pd.read_csv("diamonds.csv")


"One hot encoding for one column"

one_hot_encoder=OneHotEncoder(sparse_output=False)
encoded_data=one_hot_encoder.fit_transform(df[['color']])
print(encoded_data)

#Newly generated data frame with columns and their appropriate names

encoded_df=pd.DataFrame(encoded_data,columns=one_hot_encoder.get_feature_names_out())
print(encoded_df)
final_df=pd.concat([df,encoded_df],axis=1)
print(final_df)
final_df.drop('color',axis=1,inplace=True)
print(final_df)

"One hot encoding for more than one column"

one_hot_encoder_multiple_cols = OneHotEncoder(sparse_output=False)
encoded_data=one_hot_encoder_multiple_cols.fit_transform(df[['cut','color']])
print(encoded_data)

#Newly generated data frame with columns and their appropriate names

encode_df = pd.DataFrame(encoded_data,columns=one_hot_encoder_multiple_cols.get_feature_names_out())
print(encode_df)
final_df=pd.concat([df,encode_df],axis=1)
print(final_df)
final_df.drop(df[['cut','color']],axis=1,inplace=True)
print(final_df)