This repository contains Python code demonstrating different approaches of anamoly detection for different types of data

1. Anomaly Detection Using Z-Score Method on Iris Dataset
1-anomaly_detection_zscore_iris.py
This code demonstrates how to detect anomalies in the Iris dataset using the Z-score method. Anomalies are identified based on the deviation of data points from the mean and standard deviation of the dataset. The detected anomalies are highlighted, and an anomaly score is assigned to each data point.

2. Anomaly Detection Using Moving Average and Standard Deviation on Time Varying Dataset
2-anomaly_detection_movavg_stdev_varying_ranges.py
Given the varying ranges of the data over time, using a fixed threshold like the Z-score may not be appropriate. An alternative approach is to use a rolling window method to compute anomalies.
This code illustrates anomaly detection on a dataset with varying ranges of data over time like stock prices using the moving average and standard deviation method. Anomalies are identified based on deviations from the moving average and standard deviation of the dataset. The detected anomalies are highlighted, and an anomaly score is assigned to each data point. In the output plot, you can see that detected anamolies length varies for different time periods.

3. Anomaly Detection Using Isolation Forest on Iris Dataset
3-anomaly_detection_isolation_forest_iris.py
This code showcases anomaly detection using the Isolation Forest algorithm on the Iris dataset. The dataset used is same as 1. But here the approach is a machine learning based one.
Isolation Forest is an ensemble machine learning algorithm that isolates anomalies by randomly selecting a feature and then splitting it randomly at some point between the maximum and minimum values of the selected feature. Anomalies are identified as data points that have shorter path lengths in the tree structures, indicating they are easier to isolate.

4. Anomaly Detection Using Machine Learning on Time Varying Dataset
4-anomaly_detection_isolation_forest_varying_ranges.py
Code 2 done differently using machine learning based approach

These code samples provide practical examples of how to apply different anomaly detection techniques to various datasets, showcasing their effectiveness in identifying outliers and anomalies.

Instructions
Ensure you have Python installed on your system.
Install the required libraries listed below.
Run the Python scripts to observe anamolies.
Adjust parameters such as the window_size, threshold, etc to optimize the results.

Requirements
numpy
pandas
scipy
matplotlib
scikit-learn

License
This code is provided under the MIT License.