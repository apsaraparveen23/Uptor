import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Step 1: Create sample time series data (Temperature per day)
date_rng = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
temperature = np.random.normal(loc=30, scale=3, size=len(date_rng))  # random daily temperatures

df = pd.DataFrame({'Date': date_rng, 'Temperature': temperature})
df.set_index('Date', inplace=True)

print("Sample Data:")
print(df.head())

# Step 2: Plot the time series
plt.figure(figsize=(10,5))
plt.plot(df.index, df['Temperature'], color='orange')
plt.title("Daily Temperature Over Time")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.show()

# Step 3: Apply seasonal decomposition (Trend, Seasonality, Residual)
decomposition = seasonal_decompose(df['Temperature'], model='additive', period=7)
decomposition.plot()
plt.show()

# Step 4: Simple Moving Average
df['7-Day Average'] = df['Temperature'].rolling(window=7).mean()

plt.figure(figsize=(10,5))
plt.plot(df['Temperature'], label='Original Data', color='orange')
plt.plot(df['7-Day Average'], label='7-Day Moving Average', color='blue')
plt.legend()
plt.title("Temperature with 7-Day Moving Average")
plt.show()
