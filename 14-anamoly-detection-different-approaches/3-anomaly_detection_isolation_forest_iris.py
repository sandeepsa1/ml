import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

##### ML based approach for anamoly detection of iris dataset #####

iris = load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                       columns= iris['feature_names'] + ['target'])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(iris_df.drop(columns=['target']))

# Train isolation forest model
clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(scaled_features)

# Predict anomalies
anomaly_predictions = clf.predict(scaled_features)
anomaly_mask = anomaly_predictions == -1

# Plot
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c='blue', label='Normal')
plt.scatter(scaled_features[anomaly_mask, 0], scaled_features[anomaly_mask, 1], c='red', label='Anomaly')

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Anomaly Detection using Isolation Forest on Iris Dataset')
plt.legend()
plt.show()
