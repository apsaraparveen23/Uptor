import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

df=pd.read_csv('car_sell_dataset.csv')
print(df)

"""To print the data types of column"""
column_types=df.dtypes
print(column_types)

"""To drop a column"""
df.drop("Car Type",axis=1,inplace=True)
print(df.columns)

"""To drop the rows """
df.drop(2,axis=0,inplace=True)
print(df)

"""To read more than one column values"""
specific_column=df[['Brand','State','Transmission','Price']]
print(specific_column)

"""To print specific rows"""
data=df[0:200]
print(data)

"""Used to select rows and columns from a DataFrame by their names"""
pd.set_option("display.max_rows",None)
data=df.loc[0:1000,['Brand','Owner','State']]
print(data)

"""Returns the unique value"""
category_details=df['State'].unique()
print(category_details)

category_details=df['State'].nunique()
print(category_details)

category_details=df['State'].value_counts()
print(category_details)

"""getting a quick summary of a DataFrame."""
data_information=df.info()
print(data_information)

"""to understand the statistical summary of your dataset."""
df_information=df.describe()
print(df_information)

"""Total missing values in the entire DataFrame/Null values"""
finding_null_value=df.isnull().sum()
print(finding_null_value)

"""to count all missing/null values across an entire DataFrame"""
finding_null_values=df.isna().sum().sum()
print(finding_null_values)

"""filling of na datas"""

df[['Brand','Model Name']] = df[['Brand','Model Name']].fillna("Missing")
print(df)

"""Filling entire data frame"""
df.fillna(100,inplace=True)
print(df)


"""filling  a empty data using sklearn"""

"""Numeric column"""
si=SimpleImputer()
df[['Year','Kilometers','Price']]=si.fit_transform(df[['Year','Kilometers','Price']])
print(df)

"""one hot encoder"""

one_hot_encoder=OneHotEncoder(sparse_output=False)
encoder_data=one_hot_encoder.fit_transform(df[['Year']])
print(encoder_data)

"""Plotting"""

"""sample scatter graph plotting thr pyplot"""
"""Scatter plot for bivariate"""
x=[1,2,3,4,5]
y=[3,5,7,8,9]
plt.scatter(x,y,c=y,cmap='rainbow')
plt.colorbar()
plt.show()


"""To generate bar graph"""

df = pd.read_csv('car_sell_dataset.csv')  # Make sure your file name and path are correct

car_counts = df['Car Type'].value_counts()
print(car_counts)

car_counts.plot(kind='bar', color='coral')
plt.title('Car Type Count')
plt.xlabel('Car Type')
plt.ylabel('Number of Entries')
plt.tight_layout()
plt.show()


"""Pie chart-shows"""

df = pd.read_csv('car_sell_dataset.csv')


owner_counts = df['Owner'].value_counts()

plt.pie(owner_counts, labels=owner_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Car Type Distribution')
plt.axis('equal')  # Ensures the pie chart is a circle
plt.show()



"""Box plot"""
Year=[2000,2012,2013,2014]
plt.boxplot(Year)
plt.show()

"""Conversion of categorical data to numerical data"""

label_encoding_object=LabelEncoder()
df['Brand']=label_encoding_object.fit_transform(df['Brand'])
print(df)

"""Ordinal Encoder"""
ordinal_encoder=OrdinalEncoder()
df[['Brand','Model Name']]=ordinal_encoder.fit_transform(df[['Brand','Model Name']])
print(df)