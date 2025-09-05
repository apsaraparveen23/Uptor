import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("diamonds.csv")

"Histogram of Diamond Prices: Understanding Price Distribution"

df['price'].plot(kind = 'hist', bins=20,color='blue',edgecolor='black')
plt.xlabel("value")
plt.ylabel("frequency")
plt.title("Distribution of product price")
print(df.ndim)
plt.show()

" One-Hot Encoding a pandas Series"

encoded_df=pd.get_dummies(df['cut'],dtype=int)
print(encoded_df)
final_df=pd.concat([df,encoded_df],axis=1)
print(final_df)
