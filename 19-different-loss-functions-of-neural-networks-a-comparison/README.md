## Neural Network Loss Functions Comparison
This repository contains code to compare the performance of different loss functions in a neural network model using TensorFlow. The code evaluates various loss functions on a synthetic binary classification dataset and plots the training and validation loss over epochs.


### Dataset
The synthetic dataset consists of 1000 samples with 20 features each, generated using the make_classification function from scikit-learn. The dataset is split into training and testing sets with a ratio of 80:20.

### Neural Network Architecture
The neural network model comprises three hidden layers with 64, 32, and 16 neurons, respectively, using the ReLU activation function. The output layer consists of one neuron with a sigmoid activation function.

### Usage
1. Clone this repository to your local machine.
1. Enable virtual environment by running scripts\activate
1. Install the required dependencies.
1. Run the Python script different-loss-functions.py
1. View the generated plot to compare the loss curves for different optimizers.

### Results
The plot displays the loss curves for different optimizers over a fixed number of training epochs. By observing the plot, you can compare the convergence behavior and training efficiency of each optimizer.

### Loss Functions
1. Binary Crossentropy
1. Hinge
1. Squared Hinge
1. Mean Squared Error
1. Huber

### Results
The training and validation loss for each loss function are plotted over epochs in separate subplots for comparison.

#### Dependencies
1. Python 3.x
1. NumPy
1. Matplotlib
1. Tensorflow
1. scikit-learn

#### License
This project is licensed under the MIT License.