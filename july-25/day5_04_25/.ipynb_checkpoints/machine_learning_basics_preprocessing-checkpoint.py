import pandas as pd
# below code is read from the current dorectory
# df=pd.read_csv("diamonds.csv")
# print(df)


# below code is read from other directory(i.e.,day3_30_jun_2025)
#to read from other directory copy the path & use 'r' before the path & if \ is not given add it before the file name diamond.csv

df=pd.read_csv(r"/july-25/day5_04_25/diamonds.csv")
print(df)