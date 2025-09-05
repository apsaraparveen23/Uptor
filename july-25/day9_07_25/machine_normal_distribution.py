# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# """"creating random data"""
# normal_data=np.random.normal(50,10,100)
# """
# 1000 indicates total number of data
# 10 indicates the standard deviation
# 50 indicates mean-avg
# """
# print(normal_data)
# print(type(normal_data))
# """It tells you how many axes or directions your data has"""
# print((normal_data.ndim))# o/p is 1 which means series of data
#
# df=pd.DataFrame(normal_data,columns=["normal data"])
# print(df)
#
# print(type(df))
# """plotting this is graph"""
# df['normal data'].plot(kind='hist', bins=30, color='yellow', edgecolor='black')#histograph
# plt.xlabel("value")
# plt.ylabel("frequency")
# print(df.ndim)
# plt.show()


"""program for normal distribution"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

""""creating random data"""
normal_data=np.random.uniform(10,50,100)
"""
1000 indicates total number of data
"""
print(normal_data)
print(type(normal_data))
"""It tells you how many axes or directions your data has"""
print((normal_data.ndim))# o/p is 1 which means series of data

df=pd.DataFrame(normal_data,columns=["uniform data"])
print(df)

print(type(df))
"""plotting this is graph"""
df['uniform data'].plot(kind='hist', bins=30, color='yellow', edgecolor='black')#histograph
plt.xlabel("value")
plt.ylabel("frequency")
print(df.ndim)
plt.show()
