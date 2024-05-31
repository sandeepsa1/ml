## A multi layer Neural Network to classify images
A multi layer Neural Network designed from scratch to classify a set of images. This code snippet is not using any machine learning libraries. This was developed to familiarize with the forward and backward propagation algorithms, initializing weight and bias paramaters, how to calculate costs and derivatives and updating the weights after each iteration.</br>
Number of layers and its nodes an be defined in the variable 'layer_nodes' in 'train.py'. Hidden layers will be using RELU activation and the output should be a single node with sigmoid activation.

### Instructions
1. Clone this repository to your local machine.
1. Enable virtual environment by running scripts\activate
1. Install the required dependencies.
1. Download cats and dogs images from the below URL and extract it. Keep it inside the 'data' folder in the folder structure 'data/cats_and_dogs/training_set/' for training and 'data/cats_and_dogs/test_set/' for test.</br>
https://www.kaggle.com/datasets/tongpython/cat-and-dog
1. Run the Python scripts mentioned below.

### Script files
1. vectorize_images.py - Helper files for converting train and test images to vectorized format.
1. nn_multi_layers.py - Contains all the Neural Network functions.
1. train.py - Run this file to train the Neural Networks.
1. predict.py - Run this script after training to check the accuracy.</br>

More details inline in the code files.

### Results
The model predicts the accuracy using test data. More than 70% accuracy is achieved. Adjust number of hidden layers, the number of the hidden layer nodes, learning_rate and epochs to see the performance variations.

#### Dependencies
1. Python 3.x
1. NumPy
1. matplotlib

#### License
This project is licensed under the MIT License.