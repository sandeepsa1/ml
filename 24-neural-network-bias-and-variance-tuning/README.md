## Neural Network Bias and Variance Tuning
This repository contains two Python scripts that demonstrate how to tune neural networks to reduce bias and variance. The first script focuses on <b>reducing bias by increasing the number of layers or iterations</b>, while the second script demonstrates <b>reducing variance by increasing the number of samples or applying regularization</b>.

#### Dependencies
1. numpy
1. matplotlib
1. scikit-learn
1. tensorflow

### Scripts

#### 1. Reducing Bias by Increasing Layers and Iterations
##### File: reduce_bias.py</br>
This script demonstrates how to increasing the number of layers or nodes within each layer or increasing number of iterations. It trains a neural network on a regression dataset with a small number of layers and iterations, and then retrains it with more layers and iterations.
##### Expected Output
The script generates three plots:
1. Training and validation loss for both the initial and retrained models.
2. Data and predictions for the initial model (fewer layers/iterations).
3. Data and predictions for the retrained model (more layers/iterations) in the same plot.</br>
The R² scores are displayed on the prediction plots to show the model's performance.

#### 2. Reducing Variance by Increasing Samples and Applying Regularization
##### File: reduce_variance.py</br>
This script demonstrates how to reduce variance in a neural network by increasing the number of samples or applying L2 regularization. It trains a neural network on a smaller dataset without regularization, and then retrains it on a larger dataset with regularization.
##### Expected Output
The same plots as above.

### Example Data Generation
Both scripts generate synthetic regression data using the make_regression function from scikit-learn. This allows for a controlled demonstration of bias and variance reduction techniques.

### Model Architecture
The neural networks in both scripts are built using tensorflow and keras. The model consists of:
1. Input layer
2. Hidden layers with ReLU activation
3. Output layer with linear activation

### Evaluation Metrics
1. <b>R² Score:</b> Indicates how well the model's predictions fit the actual data.
2. <b>Train and Test Error %:</b> Shows bias and variance improvements

#### License
This project is licensed under the MIT License.