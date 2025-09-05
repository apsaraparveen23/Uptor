import pandas  as pd

df=pd.read_csv("diamonds.csv")
# print(df)
"""below two lines is to get all the columns names of the df"""
# column_names=df.columns
# print(column_names)

# column_types = df.dtypes
# print(column_types)

# print(df.head(10))
# #"""below is to get datatypes"""
#
# df_data_types=df.dtypes
# print(df_data_types)

# """below two line is to drop a column from df"""
# df.drop("Unnamed: 0",axis= 1,inplace=True)
# print(df.columns)

# """below line is to read only carat column"""
# specific_column=df['carat']
# print(specific_column)

"""to read more than one column"""
# specific_column=df[['carat','cut']]
# print(specific_column)

"""below is to fetch only the row from 0 to 200 for all colum
This process using : to fetch rows alone is called slicing"""

# data=df[0:200]
# print(data)

# """below lines will help us getting rows and columns combined"""
# data=df.loc[0:200,["carat","cut"]]
# print(data)

# category_details=df['cut'].unique()
# print(category_details)

# category_details=df['cut'].value_counts()
# print(category_details)

"""for counting the number of distinct values"""

# category_details=df['cut'].nunique()
# print(category_details)

"""getting a quick summary of a DataFrame."""
# df_information=df.info()
# print(df_information)

"""to understand the statistical summary of your dataset"""
# df_description = df.describe()
# print(df_description)