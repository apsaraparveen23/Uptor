import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

df=pd.read_csv("diamonds.csv")
# print(df)
label_encoding_object=LabelEncoder()
df['cut']=label_encoding_object.fit_transform(df['cut'])
print(df)
print(df.dtypes)
label_mapping=dict(zip(label_encoding_object.classes_,label_encoding_object.transform(label_encoding_object.classes_)))
print(label_mapping)