import pandas as pd
# below code is read from the current directory

# df=pd.read_csv("diamonds.csv")
# print(df)

# # to get the output of entire rows
pd.set_option("display.max_rows",None)

df=pd.read_csv("diamonds.csv")
print(df)

# below code is read from other directory(i.e.,day3_30_jun_2025)
# #to read from other directory copy the path & use 'r' before the path & if \ is not given add it before the file name diamond.csv

# df=pd.read_csv(r"C:\Users\apsar\PycharmProjects\PythonProject\Desktop\uptor\June_25\day3_30_jun_2025\diamonds.csv")
# print(df)

#  To print the data types of column
# column_types=df.dtypes
# print(column_types)

# df=pd.read_csv("diamonds.csv")
# print(df.head(10))

# to print last 10 rows
df=pd.read_csv("diamonds.csv")
print(df.tail(10))