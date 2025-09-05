# import matplotlib.pyplot as plt
# """sample scatter graph plotting thr pyplot"""
# """Scatter plot for bivariate"""
# x=[1,2,3,4,5]
# y=[3,5,7,8,9]
# plt.scatter(x,y,color='red')
# plt.show()

import matplotlib.pyplot as plt
"""sample scatter graph plotting thr pyplot"""
"""Scatter plot for bivariate"""
x=[1,2,3,4,5]
y=[3,5,7,8,9]

"""c represents colr.multiple coolors ge"""
plt.scatter(x,y,c=y,cmap='rainbow')
plt.colorbar()#shows number to color mapping in the side of the graph
plt.show()