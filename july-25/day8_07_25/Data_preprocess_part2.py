import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

df=pd.read_csv("diamonds.csv")
# print(df)

"""Data types for columns"""
columns=df.columns
# print(columns)
# print(df.dtypes)

"""Initializes an empty list to store column names with object-type data."""
object_columns=[]

"""To extract the object type"""

""" Loops through each column name in the list called columns (usually obtained via df.columns)."""
for col in columns:
    if df[col].dtype=='O':
        object_columns.append(col)

# print(object_columns)

"""Finding categorical data"""

# finding_categories=df['cut'].value_counts()
# print(finding_categories)
# finding_categories_unique=df['cut'].unique()
# print(finding_categories_unique)
#
# finding_categories=df['cut'].nunique()
# print(finding_categories)

#
# finding_categories=df['cut'].value_counts()
# print(finding_categories)

# cut_categories_list=df['cut'].unique()
# print(cut_categories_list)

# df.dropna(inplace=True)
# df.reset_index(inplace=True)
#
# cut_categories_list=df['cut'].unique()
# print(cut_categories_list)

# df['cut']=df['cut'].map({'':23,'Ideal':1,'Premium':4,'Good':5,'Very Good':3,'Fair':8})
# print(df)
# print(df.dtypes)

"""Conversion of categorical data to numerical data"""

# label_encoding_object=LabelEncoder()
# df['cut']=label_encoding_object.fit_transform(df['cut'])
# print(df)

"""converting categorical data to  numerical data for more than one column"""
"""Note:below code will throw error due to two columns in one line,we should handle separtely"""


# label_encoding_object=LabelEncoder()
# df[['cut','color']]=label_encoding_object.fit_transform(df['cut','color'])
# print(df)

"""""converting from int to float for one columns """""
# ordinal_encoder=OrdinalEncoder()
# df['cut']=ordinal_encoder.fit_transform(df[['cut']])
# print(df)

"""converting from int to float for two columns """
# ordinal_encoder=OrdinalEncoder()
# df[['cut','color']]=ordinal_encoder.fit_transform(df[['cut','color']])
# print(df)


unique_data_post_conversion=df['cut'].unique()
print(unique_data_post_conversion)
print(df.dtypes)