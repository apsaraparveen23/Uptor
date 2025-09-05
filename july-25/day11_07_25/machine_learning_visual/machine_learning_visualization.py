import matplotlib.pyplot as plt

"""sample scatter graph plotting thr pyplot"""
"""Scatter plot for bivariate"""
# x=[1,2,3,4,5]
# y=[3,5,7,8,9]

"""c represents colr.multiple coolors ge"""
# plt.scatter(x,y,c=y,cmap='rainbow')
# plt.colorbar()#shows number to color mapping in the side of the graph
# plt.show()

"""Below code is using seaborn library to generate graph for visualization"""

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

"""X,Y graph"""

"""sample scatter graph plotting thr pyplot"""
"""Scatter plot for bivariate"""
# x=[1,2,3,4,5]
# y=[3,5,7,8,9]
# df=pd.DataFrame({"key1":x,"key2":y})

"""creating dict for extra leraning"""
# df=pd.DataFrame(dict([("key1",x),("key2",y)]))
# print(df)

"""x and y should be column names (as strings) from your DataFrame,
not variables containing data."""
"""You need to explicitly pass the data= argument with the DataFrame (df) that holds those columns."""
"""we can't just give x,y as pandas"""

# sns.scatterplot(x='key1',y='key2',data=df)

"""can use this below code instead of line 29"""
# sns.scatterplot(df,x='key1',y='key2')



"""Even if use Seaborn to generate graph we use plt.show() from matplotlib"""
# plt.show()

"""Bar chart"""
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# team=['csk','rcb','mi','csk','mi','kkr']
# trophy=[1,1,17,11,1,2]
# """To genarate bar graph ( i.e.,category vs numeric),adds(summation or aggregation) and plots the value"""
# plt.bar(team,trophy)
# plt.show()

"""Pie chart-shows the percentage for each category"""
# import matplotlib.pyplot as plt
# import numpy as np
#
# team=['csk','rcb','mi','kkr']
# trophy=np.array([1,17,4,11])


"""category vs count for pie chart"""
"""syntax to display with percentage autopct='%1.1f%%'"""
# plt.pie(trophy,labels=team,autopct='%1.1f%%')
# plt.show()

"""Finding outlier through boxplot"""
# import matplotlib.pyplot as plt
# import numpy as np
#
# age=[3,5,6,7,70,8,60]
# plt.boxplot(age)
# plt.show()