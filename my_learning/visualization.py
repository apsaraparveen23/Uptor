# import pandas as pd
# import matplotlib.pyplot as plt
#
# df=pd.read_csv("diamonds.csv")
# print(df.columns)
#
# "Scatter Plot"
# sample_df = df.sample(1000, random_state=42)  # Pick 1000 random rows
# plt.xlabel('carat')
# plt.ylabel('price')
# plt.scatter(sample_df['carat'], sample_df['price'], color='blue', alpha=0.6)
# plt.show()
#
# "Bar plot"
# avg_price_by_cut = df.groupby('cut')['price'].mean().sort_values()
# avg_price_by_cut.plot(kind='bar', color='purple')
# plt.xlabel('Cut')
# plt.ylabel('Average Price')
# plt.title('Average Diamond Price by Cut')
# plt.xticks(rotation=45)
# plt.show()


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# "Bar chart"
#
# df=pd.read_csv("diamonds.csv")
# sns.countplot(x='cut', hue='cut', data=df, palette='pastel', legend=False)
# plt.title('Count of Diamonds by Cut')
# plt.xlabel('Cut Quality')
# plt.ylabel('Count')
# plt.show()
#
# "Scatter plot"
#
# df=pd.read_csv("diamonds.csv")
# sns.scatterplot(x='carat', y='price', data=df, alpha=0.5)
# plt.title('Diamond Price vs Carat')
# plt.xlabel('Carat')
# plt.ylabel('Price')
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Programming_languages= ['Python', 'JavaScript', 'C++', 'Java']
sizes = np.array([50, 25, 15, 10])

# Create pie chart
plt.pie(sizes,labels=Programming_languages,autopct='%1.1f%%')
plt.title('Favorite Programming Languages', fontsize=14)

# Show plot
plt.show()


"""Finding outlier through boxplot"""

expenses = [25, 27, 30, 28, 26, 29, 100]

# Create boxplot
plt.boxplot(expenses)

# Add title and labels
plt.title("Monthly Expenses Boxplot")
plt.ylabel("Expense")

# Show plot
plt.show()
