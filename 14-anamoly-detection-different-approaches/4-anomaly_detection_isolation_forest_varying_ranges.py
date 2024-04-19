import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

##### ML based approach for time series data. #####

# Generate sample data with varying ranges
np.random.seed(0)
dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
values = np.random.normal(loc=0, scale=1, size=len(dates)) * np.arange(1, len(dates) + 1) ** 0.5

# Add anomalies
values[100:110] += 10
values[200:210] -= 8

data = pd.DataFrame({'Date': dates, 'Value': values})

# Feature engineering
data['Dayofyear'] = data['Date'].dt.dayofyear

# Train Isolation Forest model
clf = IsolationForest(contamination=0.08, random_state=42)
clf.fit(data[['Value', 'Dayofyear']])

# Predict anomalies
data['Anomaly'] = clf.predict(data[['Value', 'Dayofyear']])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Value'], label='Original Data', color='blue')
plt.scatter(data.loc[data['Anomaly'] == -1, 'Date'], data.loc[data['Anomaly'] == -1, 'Value'], 
            color='red', label='Anomalies')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Anomaly Detection using Isolation Forest on time varying data')
plt.legend()
plt.show()
