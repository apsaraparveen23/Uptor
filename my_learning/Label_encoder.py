import pandas as pd

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

df = pd.read_csv("diamonds.csv")

""" Label encoding for one columns"""

label_encoding = LabelEncoder()
df['cut'] = label_encoding.fit_transform(df['cut'])
print(df)

""" Label encoding for two columns"""

for col in ['cut','color']:
    le=LabelEncoder()
    df[col + '_encoded_data']=le.fit_transform(df[col])
    print(df)

"""Label encoding  and decoding for one column"""

le=LabelEncoder()
df['cut_encoded']=le.fit_transform(df['cut'])

df['cut_decoded']=le.inverse_transform(df['cut_encoded'])
print(df)

"""Label encoding  and decoding for two  column"""

#Columns to encode

cols_to_encode = ['cut', 'color']
encoders = {}

#Encode each column
for col in cols_to_encode:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    encoders[col] = le

# Decode each column
for col in cols_to_encode:
    encoded_col = col + '_encoded'
    decoded_col = col + '_decoded'
    df[decoded_col] = encoders[col].inverse_transform(df[encoded_col])
print(df[['cut', 'cut_encoded', 'cut_decoded', 'color', 'color_encoded', 'color_decoded']])


