import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1️⃣ Read the dataset
df = pd.read_csv("diamonds.csv")
print(df.columns)

# 2️⃣ Scatter Plot (Matplotlib)
sample_df = df.sample(1000, random_state=42)
plt.figure(figsize=(8, 6))
plt.scatter(sample_df['carat'], sample_df['price'], color='blue', alpha=0.6)
plt.xlabel('Carat')
plt.ylabel('Price')
plt.title('Diamond Price vs Carat (Matplotlib Scatter Plot)')
plt.show()

# 3️⃣ Bar Plot (Average Price by Cut - Matplotlib)
avg_price_by_cut = df.groupby('cut')['price'].mean().sort_values()
plt.figure(figsize=(8, 6))
avg_price_by_cut.plot(kind='bar', color='purple')
plt.xlabel('Cut')
plt.ylabel('Average Price')
plt.title('Average Diamond Price by Cut (Matplotlib Bar Plot)')
plt.xticks(rotation=45)
plt.show()

# 4️⃣ Bar Chart (Count of Diamonds by Cut - Seaborn)
plt.figure(figsize=(8, 6))
sns.countplot(x='cut', hue='cut', data=df, palette='pastel', legend=False)
plt.title('Count of Diamonds by Cut (Seaborn Count Plot)')
plt.xlabel('Cut Quality')
plt.ylabel('Count')
plt.show()

# 5️⃣ Scatter Plot (Price vs Carat - Seaborn)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='carat', y='price', data=df, alpha=0.5, color='green')
plt.title('Diamond Price vs Carat (Seaborn Scatter Plot)')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.show()

# 6️⃣ Pie Chart (Programming Languages)
Programming_languages = ['Python', 'JavaScript', 'C++', 'Java']
sizes = np.array([50, 25, 15, 10])
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=Programming_languages, autopct='%1.1f%%', startangle=90)
plt.title('Favorite Programming Languages', fontsize=14)
plt.show()

# 7️⃣ Boxplot (Finding Outliers)
expenses = [25, 27, 30, 28, 26, 29, 100]
plt.figure(figsize=(6, 5))
plt.boxplot(expenses, patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title("Monthly Expenses Boxplot")
plt.ylabel("Expense")
plt.show()
