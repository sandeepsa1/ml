## My Projects in Machine Learning and Deep Learning
This repository contains my AI learning projects.
Each folder within the repository are separate virtual environments representing a separate code implementation, accompanied by a README.md file explaining its functionalities.

## Table of Contents
- [01-perceptron-predict-within-a-square](#01-perceptron-predict-within-a-square)
- [02-perceptron-predict-within-a-circle](#02-perceptron-predict-within-a-circle)
- [03-linear-regression](#03-linear-regression)
- [04-recognize-digits-using-sigmoid-perceptron_without-ml-libraries](#04-recognize-digits-using-sigmoid-perceptron_without-ml-libraries)
- [05-predicting-random-points-within-a-circle-using-sigmoid_without-ml-libraries](#05-predicting-random-points-within-a-circle-using-sigmoid_without-ml-libraries)
- [06-digit-recognition-using-ml-library-scikit-learn](#06-digit-recognition-using-ml-library-scikit-learn)
- [07-digit-recognition-using-neural-network-library-tensorflow](#07-digit-recognition-using-neural-network-library-tensorflow)
- [08-sentiment-analysis-using-rnn](#08-sentiment-analysis-using-rnn)
- [09-image-transformations-and-matrices](#09-image-transformations-and-matrices)
- [10-understanding-image-compression-using-svd](#10-understanding-image-compression-using-svd)
- [11-reinforcement-learning-q-learning](#11-reinforcement-learning-q-learning)
- [12-ensemble-learning-methods-performance-comparison](#12-ensemble-learning-methods-performance-comparison)
- [13-clustering-kmeans_hierarchical_dbscan](#13-clustering-kmeans_hierarchical_dbscan)
- [14-anamoly-detection-different-approaches](#14-anamoly-detection-different-approaches)
- [15-transfer-learning-images](#15-transfer-learning-images)
- [16-transfer-learning_nlp](#16-transfer-learning_nlp)
- [17-reinforcement-learning-using-qtable-and-neural-network](#17-reinforcement-learning-using-qtable-and-neural-network	)

## Details
### 01-perceptron-predict-within-a-square
How to integrate perceptrons: When predicting whether a random point in a plane lies within a closed space, like a square. The approach involves combining four perceptrons to achieve accurate predictions. A simple javascript code that predicts if the points are within or outside a random square.

### 02-perceptron-predict-within-a-circle
Predicting whether a random point in a plane lies within a closed space, like a circle. The approach involves combining multiple perceptrons to achieve accurate predictions. Idea is to keep the circular space as a set of lines by drawing tangents and perpendiculars to it. Area coming under these lines are part of the circle. More number of perpendiculars gives a well defined circular space. This way prediction can be done on any type of non-linear spaces.

### 03-linear-regression
A simple linear regression sample to understand the concepts.

### 04-recognize-digits-using-sigmoid-perceptron_without-ml-libraries
A neural network to recognize digit using Sigmoid activation function. This code snippet is not using any machine learning libraries. Working on this code snippet helped me to learn the SGD and back propagation algorithms.

### 05-predicting-random-points-within-a-circle-using-sigmoid_without-ml-libraries
Predicting whether a random point in a plane lies within a closed space, like a circle. This was implemented earlier using perceptrons. This code employs a sigmoid activation function with a neural network structure of layers [2, 30, 2]. Input features include x and y coordinates, and the output classifies points as inside or outside the circle. This modification allows for a more nuanced representation, overcoming the linear separation constraint. Here np Neural network libraries are used. The SGD and the back propogation logic is implemented in the code itself.

### 06-digit-recognition-using-ml-library-scikit-learn
Digit recognition using the scikit-learn machine learning library.

### 07-digit-recognition-using-neural-network-library-tensorflow
Digit recognition using the neural network library tensorflow.

### 08-sentiment-analysis-using-rnn
Implementing a sentiment analysis tool using a simple Recurrent Neural Network (RNN). This tool aims to classify reviews from the IMDB database as either positive or negative.

### 09-image-transformations-and-matrices
This project implements image transformations of translation, rotation, scaling, and shearing on a 2D shape along the X and Y axes.

### 10-understanding-image-compression-using-svd
This repository provides a sample code demonstrating the use of Singular Value Decomposition (SVD) for image compression.

### 11-reinforcement-learning-q-learning
Trying t get basic understanding of Q-Learning algorithm using OpenAI GYM Frozen lake program.

### 12-ensemble-learning-methods-performance-comparison
This project compares the performance of different Ensemble Learning Techniques on two different datasets: the IRIS dataset and the Wine Quality dataset. The code trains three classifiers (Random Forest, AdaBoost, and Gradient Boosting) on each dataset and evaluates their accuracy and training time.

### 13-clustering-kmeans_hierarchical_dbscan
This repository contains Python code demonstrating the use of K-means clustering algorithm along with methods to determine the optimal number of clusters using the elbow method and silhouette analysis. Also implements Hierarchical and DBScan Clustering codes.

### 14-anamoly-detection-different-approaches
Detect anamolies on normal and time varying data using z-score, moving average and ML based approaches.

### 15-transfer-learning-images
Implement transfer learning on Convolutional Networks which does image recognition and other tasks.

### 16-transfer-learning_nlp
Implement transfer learning on NLP using pre-trained word embeddings. This code does Sentiment Analysis.

### 17-reinforcement-learning-using-qtable-and-neural-network
This repository contains Python scripts implementing reinforcement learning algorithms for training a blue dot to navigate towards a fixed red dot in a 10 by 10 space. The blue dot aims to reach the position of the red dot to receive maximum reward. The distance between the blue and red dots is calculated after each step, and rewards are adjusted accordingly. The goal is to maximize the reward by reaching the red dot within a maximum number of iterations. Implemented in two ways. Using Q-Table to store state-action values and also using Neural Network Q-Learning