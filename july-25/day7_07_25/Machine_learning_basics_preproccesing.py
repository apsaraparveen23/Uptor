import pandas as pd


df=pd.read_csv("diamonds.csv")
# print(df)

"""shows null values for each column"""
# finding_null_value=df.isnull().sum()
# print(finding_null_value)

"""To show the sum of all null values"""
# finding_null_value=df.isnull().sum().sum()
# print(finding_null_value)

# finding_null_value=df.isna().sum()
# print(finding_null_value)
#
# finding_null_value=df.isna().sum().sum()
# print(finding_null_value)

"""filling of na datas"""
# df['carat'] = df['carat'].fillna("hello")
# print(df)
"""Filling more than one column"""
# df[['carat','cut']] = df[['carat','cut']].fillna("hello")
# print(df)

"""Filling entire data frame"""
# df.fillna(2000,inplace=True)
# print(df)
"""using mean of specific column to fill NA in the data frame"""
# df.fillna(df['carat'].mean(),inplace=True)#mean only for numeric data
# print(df)


"""rounding of carat column"""
# df["carat"]=df['carat'].round()
# print(df)

"""rounding of to 2 digit"""
# df["carat"]=df['carat'].round(2)
# print(df)

# df['Unnamed: 0']=df['Unnamed: 0'].astype(float)
# print(df)

"""filling  a empty data using sklearn"""
from sklearn.impute import SimpleImputer

"""columns with numeric data"""
si=SimpleImputer()
df['carat']=si.fit_transform(df[['carat']])
print(df)

""" filling null column with values which occurs repeatedly within the data fram"""

si=SimpleImputer(strategy="most_frequent")
df['carat']=si.fit_transform(df[['carat']])
print(df)

"""explicitly fill missing values with a constant, in this case 200."""

si=SimpleImputer(strategy="constant",fill_value=200)
df['carat']=si.fit_transform(df[['carat']])
print(df)

"""below three lines will throw error as  simple imputer can be applied only for numeric columns"""

# si=SimpleImputer(strategy="constant",fill_value=200)
# df['cut']=si.fit_transform(df[['cut']])
# print(df)

si=SimpleImputer(strategy='constant',fill_value='unknoown')
df['cut']=si.fit_transform(df[['cut']]).squeeze()
print(df)


"""if any column is null then corresponding entire row will be dropped"""
# df.dropna(inplace=True)
# print(df)
"""to making the indexing in correct position after dropping(line 63 &64)"""
# df.reset_index(inplace=True)
# print(df)

"""below code will throw error index can happen only after drop(64 &67),it cant happen in one single line"""
# df.dropna(inplace=True).reset_index(inplace=True)
# print(df)

"""finding duplicates"""
# finding_duplicate=df[df.duplicated()]
# print(finding_duplicate)

"""dropping duplicate"""
# df.drop_duplicates(inplace=True)
# print(df)
# finding_duplicate=df[df.duplicated()]
# print(finding_duplicate)

"""dropping the first and retaining lastone"""
# df.drop_duplicates(keep='last',inplace=True)
# print(df)

#
# finding_duplicate=df[df.duplicated()]
# print(finding_duplicate)

"""dropping the original & duplicates"""

# df.drop_duplicates(keep=False,inplace=True)
# print(df)
#
# finding_duplicate=df[df.duplicated()]
# print(finding_duplicate)


