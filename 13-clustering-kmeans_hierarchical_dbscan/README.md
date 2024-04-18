1, 2, 3, - K-means Clustering with Optimal Number of Clusters Selection
4, 5 - Hierarchical and DBScan Clustering

Overview
This repository contains Python code demonstrating the use of K-means clustering algorithm along with methods to determine the optimal number of clusters using the elbow method and silhouette analysis (1, 2).
3 demonstrates K-means clustering on iris dataset
4 and 5 demonstrates Hierarchical and DBScan Clustering methods on the iris dataset to see a comparison of performances.

Code Samples
1. kmeans_clustering.py
This Python script demonstrates how to perform K-means clustering using the scikit-learn library. It generates synthetic data and applies K-means clustering to identify clusters. The clusters and cluster centers are visualized using matplotlib.

2. optimal_k_selection.py
This Python script extends the functionality of the kmeans_clustering.py script by including methods to determine the optimal number of clusters (k) using the elbow method and silhouette analysis. The elbow method helps to identify the k value based on the within-cluster sum of squares (WCSS), while silhouette analysis provides a measure of cluster separation and cohesion.

3. kmeans_iris.py
This Python script demonstrates K-Means clustering using real data of iris dataset. K-Means is a popular clustering algorithm that partitions data into K clusters based on similarity.

4. hierarchical_iris.py
This Python script demonstrates hierarchical clustering using the Iris dataset. Hierarchical clustering is a method of cluster analysis that builds a hierarchy of clusters. It does not require a pre-specified number of clusters, which makes it particularly useful for exploratory data analysis.

5. dbscan_iris.py
This Python script demonstrates DBSCAN (Density-Based Spatial Clustering of Applications with Noise) using the Iris dataset. DBSCAN is a clustering algorithm that groups together closely packed points based on density.

Instructions
Ensure you have Python installed on your system.
Install the required libraries listed below.
Run the Python scripts to observe K-means, Hierarchical and DBScan clustering results and optimal k selection.
Adjust parameters such as the number of clusters, dataset properties, and visualization settings as needed.

Requirements
scikit-learn
numpy
matplotlib
pandas

License
This code is provided under the MIT License.