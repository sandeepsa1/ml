import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##### For time varying data like stock prices, moving average and standard deviation based approach suits. #####
##### In the output plot, you can see that detected anamolies length varies for different time periods. #####

# Generate sample data with varying ranges
np.random.seed(0)
dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
values = np.random.normal(loc=0, scale=1, size=len(dates)) * np.arange(1, len(dates) + 1) ** 0.5

# Add anomalies
values[100:110] += 10
values[200:210] -= 8

data = pd.DataFrame({'Date': dates, 'Value': values})

# Calculate rolling mean and standard deviation
window_size = 30  # Adjust window size as needed
data['Rolling_Mean'] = data['Value'].rolling(window=window_size).mean()
data['Rolling_Std'] = data['Value'].rolling(window=window_size).std()

# Calculate z-score for anomaly detection
data['Z_Score'] = (data['Value'] - data['Rolling_Mean']) / data['Rolling_Std']
threshold = 2  # Adjust threshold as needed

# Detect anomalies based on z-score
data['Anomaly'] = np.abs(data['Z_Score']) > threshold

# Plot
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Value'], label='Original Data', color='blue')
plt.scatter(data.loc[data['Anomaly'], 'Date'], data.loc[data['Anomaly'], 'Value'], 
            color='red', label='Anomalies')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Anomaly Detection using Moving Average and Standard Deviation on time varying data')
plt.legend()
plt.show()