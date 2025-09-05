import pandas as pd

"""Reads the CSV file and stores the data in a variable called df"""
df=pd.read_csv("diamonds.csv")
print(df)

"""Reads the CSV and immediately prints the data without storing it"""
print(pd.read_csv("diamonds.csv"))

"""The set_option  in pandas tells not to limit the number of rows shown when you print a DataFrame."""
"""Useful in data cleaning or debugging, where you need to see every record."""
"""Need to place the pd.line set option line after import statements"""

pd.set_option("display.max_rows",None)
df=pd.read_csv("diamonds.csv")
print(df)

"""To read file from other directory"""""
df=pd.read_csv(r"C:\Users\apsar\PycharmProjects\PythonProject\Desktop\uptor\july-25\day5_04_25\diamonds.csv")
print(df)

"""To print the data types of column"""
column_types=df.dtypes
print(column_types)

"""To print the first 10 rows"""
print(df.head(10))

"""To print last 10 rows"""
print(df.tail(10))

"""To get the names of column"""
column_names=df.columns
print(column_names)

"""To drop a column"""
df.drop("Unnamed: 0",axis=1,inplace=True)
print(df.columns)

df.drop("depth",axis=1,inplace=True)
print(df.columns)

"""To drop the rows use axis=0"""
df.drop(2,axis=0,inplace=True)
print(df)

"""To read one column values"""
specific_column=df['carat']
print(specific_column)

"""To read more than one column values"""
specific_column=df[['carat','cut','x']]
print(specific_column)

"""This line is slicing your DataFrame to show only the first 200 rows."""
data=df[0:200]
print(data)

"""To print data from 30 to 51 from top to bottom"""
data = df[30:51]
print(data)

"""To get last 20 rows"""
data=df[-20:]
print(data)

"""To print data for specific rows from bottom to top"""
range_data=df[-50:-30]
print(range_data)

"""Used to select rows and columns from a DataFrame by their names"""
pd.set_option("display.max_rows",None)
data=df.loc[0:200,['carat','cut','x','y']]
print(data)

"""Returns the unique value"""
category_details=df['cut'].unique()
print(category_details)

"""count how often each unique value appears in a column."""
category_details=df['cut'].value_counts()
print(category_details)

"""for counting the number of distinct values"""
category_details=df['cut'].nunique()
print(category_details)

"""getting a quick summary of a DataFrame."""
data_information=df.info()
print(data_information)

"""to understand the statistical summary of your dataset."""
df_information=df.describe()
print(df_information)