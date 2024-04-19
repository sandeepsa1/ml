import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.stats import zscore

##### Z-score based approach works good for data that doesn't depend on time #####

iris_sklearn = load_iris()
iris_data = pd.DataFrame(data=iris_sklearn.data, columns=iris_sklearn.feature_names)

z_scores = zscore(iris_data)
threshold = 2 # Change this and test

anomaly_indices = np.any(np.abs(z_scores) > threshold, axis=1)

# Filter out the anomalies and show it in csv file
#anomaly_data = iris_data[anomaly_indices]
#anomaly_data.to_csv('iris_anomalies.csv', index=False)

# Plot normal data in blue colour and anamolies in red colour
anomalies = iris_data[anomaly_indices]
normal_data = iris_data[~anomaly_indices]
plt.scatter(normal_data.iloc[:, 0], normal_data.iloc[:, 1], c='blue', label='Normal Data')
plt.scatter(anomalies.iloc[:, 0], anomalies.iloc[:, 1], c='red', label='Anomalies')

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Anomaly Detection using Z-Score on Iris Dataset')
plt.legend()
plt.show()