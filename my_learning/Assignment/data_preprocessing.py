import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("diamonds.csv")
print("\nInitial Data Loaded:")
print(df.head())

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nBasic Summary:")
print(df.describe())

print("\nDataFrame Info:")
df.info()

print("\nChecking Null Values:")
print(df.isnull().sum())

numeric_columns = ['carat', 'depth', 'price']
si_mean = SimpleImputer(strategy='mean')
df[numeric_columns] = si_mean.fit_transform(df[numeric_columns])

categorical_columns = ['cut', 'color']
si_cat = SimpleImputer(strategy='most_frequent')
for col in categorical_columns:
    df[col] = si_cat.fit_transform(df[[col]]).ravel()

print("\nAfter Imputation:")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"\nShape after removing duplicates: {df.shape}")

if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

print("\nColumns after dropping unnecessary ones:")
print(df.columns)

if 'carat' in df.columns:
    df['carat'] = df['carat'].round(2)

if 'depth' in df.columns:
    df['depth'] = df['depth'].astype(float)

print("\nAfter rounding and type conversion:")
print(df.head())

label_encoders = {}
categorical_columns = ['cut', 'color']

for col in categorical_columns:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

print("\nEncoded Columns Added:")
print(df[[col + '_encoded' for col in categorical_columns]].head())

for col in categorical_columns:
    encoded = col + '_encoded'
    decoded = col + '_decoded'
    df[decoded] = label_encoders[col].inverse_transform(df[encoded])

print("\nEncoding/Decoding Example:")
print(df[['cut', 'cut_encoded', 'cut_decoded', 'color', 'color_encoded', 'color_decoded']].head())

print("\nFinal DataFrame Info:")
df.info()

print("\nFinal Cleaned Data (first 10 rows):")
print(df.head(10))
