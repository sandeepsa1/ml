import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

iris_df = pd.DataFrame(data=X, columns=iris.feature_names)
iris_df['target'] = y

# Compute the linkage matrix
linkage_matrix = linkage(X, method='ward')

# Plot
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, labels=y)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Iris Data Samples')
plt.ylabel('Distance')
plt.show()