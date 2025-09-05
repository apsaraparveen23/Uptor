import pandas as pd
df=pd.read_csv("diamonds.csv")
col_name=df.columns
print(col_name)
print(df.columns.tolist())
print(df.columns)
# import matplotlib.pyplot as plt
#
# # Load the CSV file
# df = pd.read_csv('car_sell_dataset.csv')  # Update path if needed
#
# # Count how many times each car type appears
# owner_counts = df['Owner'].value_counts()
#
# # Plot a pie chart
# plt.pie(owner_counts, labels=owner_counts.index, autopct='%1.1f%%', startangle=140)
# plt.title('Car Type Distribution')
# plt.axis('equal')  # Ensures the pie chart is a circle
# plt.show()
#
