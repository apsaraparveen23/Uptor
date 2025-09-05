import pandas as pd

from sklearn.impute import SimpleImputer
df=pd.read_csv("diamonds.csv")
print(df)

"""Total missing values in the entire DataFrame/Null values"""

finding_null_value=df.isnull().sum()
print(finding_null_value)

"""To show the sum of all null values"""

finding_null_values=df.isnull().sum().sum()
print(finding_null_values)

"""To detect missing values like nan"""

finding_null_values=df.isna().sum()
print(finding_null_values)

"""To count all missing/null values across an entire DataFrame"""

finding_null_values=df.isna().sum().sum()
print(finding_null_values)

"""Filling of na datas"""

df['carat'] = df['carat'].fillna("hello")
print(df)

"""Filling na datas for more than one column"""

df[['carat','cut']] = df[['carat','cut']].fillna("hello")
print(df)

"""Filling entire data frame"""

df.fillna(2000,inplace=True)
print(df)

"""using mean of specific column to fill NA in the data frame"""

df.fillna(df['carat'].mean(),inplace=True)
print(df)

"""Rounding of carat column"""

round_off=df['carat'].round()
print(round_off)

"""rounding to 2 digit"""

round_off_2digit=df['carat'].round(2)
print(round_off_2digit)

"""rounding of numeric value to float"""

df['Unnamed: 0']=df['Unnamed: 0'].astype(float)
print(df)

"""Renaming columns"""

df.rename(columns={'carat':'stone_mass'}, inplace=True)
print(df)

"""Replacing column values"""

df.replace({'cut':{'Fair':'Poor','Ideal':'Excellent'}},inplace=True)
print(df)
print(df['cut'].unique())





"""Filling  a empty data using sklearn"""


"""Filling the missing values for numeric column"""
si=SimpleImputer()
df['carat']=si.fit_transform(df[['carat']])
print(df)

"""Filling null column with values which occurs repeatedly within the data frame"""
si=SimpleImputer(strategy="most_frequent")
df['carat']=si.fit_transform(df[['carat']])
print(df)

"""Explicitly fill missing values with a constant"""
si =SimpleImputer(strategy="constant",fill_value=700)
df['carat']=si.fit_transform(df[['carat']])
print(df)

"""Filling Missing Values with the Median Using SimpleImputer"""
si =SimpleImputer(strategy="median")
df['carat']=si.fit_transform(df[['carat']])
print(df)

"""Filling Missing Values with the Mean Using SimpleImputer"""
si =SimpleImputer(strategy="mean")
df['carat']=si.fit_transform(df[['carat']])
print(df)

"Imputing multiple columns"

si=SimpleImputer(strategy='median')
df[['carat','depth','price']]=si.fit_transform(df[['carat','depth','price']])
print(df)

"""or (passing the values by creating a list)"""

numeric_columns=['carat','depth','price']#creating list
si=SimpleImputer(strategy='mean')
df[numeric_columns]=si.fit_transform(df[numeric_columns])
print(df)


"""Handling  simple imputer with categorical data"""

"""Handling Missing Values in 'cut' Column Using SimpleImputer(by using ravel)"""

si=SimpleImputer(strategy='most_frequent')
df['cut']=si.fit_transform(df[['cut']]).ravel()
print(df)

"""Handling Missing Values in 'cut' Column Using SimpleImputer(by using squeeze)"""
"""ravel or squeeze can be used because after applying simple imputer 
o/p will be in 2d array but panadas need 1D array series"""

si=SimpleImputer(strategy='most_frequent')
df['cut']=si.fit_transform(df[['cut']]).squeeze()
print(df)

si=SimpleImputer(strategy='constant',fill_value='unknown')
df['cut']=si.fit_transform(df[['cut']]).squeeze()
print(df)


"""error thrown since used a categorical col,Simple imputer is applicable for numeric  col"""
# si=SimpleImputer(strategy="constant",fill_value=2000)
# df['cut']=si.fit_transform(df[['cut']])
# print(df)


"Drops rows with NaN values"

df.dropna(inplace=True)
print(df)

"Resetting the index"

df.reset_index(inplace=True)
print(df)

"Checking for duplicate rows in the data frame"

finding_duplicates=df[df.duplicated()]
print(finding_duplicates)

"Removes exact duplicate rows"

df.drop_duplicates(inplace=True)
print(df)
finding_duplicates=df[df.duplicated()]#checks again for duplicates after dropping
print(finding_duplicates)

"Drops the duplicate rows but retains the last occurrence of each duplicate "

df.drop_duplicates(keep='last',inplace=True)
print(df)
finding_duplicates=df[df.duplicated()]
print(finding_duplicates)

"""Both original and duplicate rows are dropped"""

df.drop_duplicates(keep=False,inplace=True)
print(df)
finding_duplicates=df[df.duplicated()]
print(finding_duplicates)

