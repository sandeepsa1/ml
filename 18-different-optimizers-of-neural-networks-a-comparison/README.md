## Neural Network Optimizers Comparison
This repository contains Python scripts to compare the performance of different optimizers using TensorFlow and Keras. The first code generates random data, defines a simple neural network model with one input and one output neuron, and then trains the model using different optimizers such as SGD, Adam, RMSprop, Adagrad, and Adadelta. Finally, it plots the loss curves for each optimizer to compare their performance.

The second code uses a synthetic dataset and defines a neural network model with multiple hidden layers. Then does the performance comparison for different optimizers and plots the results. In this code sample, it is able to observe visible differences in performance among different optimizers. Adjust the model architecture and hyperparameters as needed to see more significant differences.


### Usage
1. Clone this repository to your local machine.
1. Enable virtual environment by running scripts\activate
1. Install the required dependencies.
1. Run the Python script optimizer_comparison.py.
1. View the generated plot to compare the loss curves for different optimizers.

### Results
The plot displays the loss curves for different optimizers over a fixed number of training epochs. By observing the plot, you can compare the convergence behavior and training efficiency of each optimizer.

### Optimizers Included
1. Stochastic Gradient Descent (SGD)
1. Adam
1. RMSprop
1. Adagrad
1. Adadelta

### Conclusion
Based on the loss curves, analyze and choose the optimizer that best suits your specific neural network architecture and training dataset.

#### Dependencies
1. Python 3.x
1. NumPy
1. Matplotlib
1. Tensorflow
1. scikit-learn

#### License
This project is licensed under the MIT License.