## A Neural Network to classify images as dogs or not
A 2 layer Neural Network designed to classify a set of images as dogs or not. This code snippet is not using any machine learning libraries. This was developed to familiarize with the forward and backward propagation algorithms, initializing weight and bias paramaters, how to calculate costs and derivatives and updating the weights after each iteration.</br>
Code uses a 2 layer neural network. Input layer has 12288 nodes (64, 64, 3) and hidden layer is of 20 nodes with RELU activation and the output is a single node with sigmoid activation.

### Instructions
1. Clone this repository to your local machine.
1. Enable virtual environment by running scripts\activate
1. Install the required dependencies.
1. Download cats and dogs images from the below URL and extract it. Keep it inside the 'data' folder in the folder structure 'data/cats_and_dogs/training_set/' for training and 'data/cats_and_dogs/test_set/' for test.</br>
https://www.kaggle.com/datasets/tongpython/cat-and-dog
1. Run the Python scripts mentioned below.

### Script files
1. vectorize_images.py - Helper files for converting train and test images to vectorized format.
1. nn_2layers.py - Contains all the Neural Network functions.
1. train.py - Run this file to train the Neural Networks.
1. predict.py - Run this script after training to check the accuracy.</br>

More details inline in the code files.

### Results
The model predicts the accuracy using test data. More than 80% accuracy is achieved. Adjust number of layers or the count of the hidden layers, learning_rate and epochs to see the performance variations.

#### Dependencies
1. Python 3.x
1. NumPy
1. matplotlib

#### License
This project is licensed under the MIT License.