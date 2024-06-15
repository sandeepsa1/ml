## Finding the best set of hyper parameters for a Neural Network model
Finding the best set of hyper parameters is crucial for effectively training neural networks. The most important hyper parameters that we need to tune to get a good performing model  are:
1. Learning Rate
2. Number of Hidden Layers needed
3. Number of nodes within each Hidden Layer
4. Mini Batch Size for gradient descent
5. Applying Regularizations like L1, L2 and Dropout

This code repository has scripts to find a suitable configuration of the above set of hyper parameters.</br>
There are also some other important hyper parameters which can be used to imporve the performance of a NN model.</br>
Some of them like Learning Rate Decay, Beta values of optimizers, choosing the activation functions/cost functions/optimizers can be included in the code by making some modifications.</br>
There are also some parameters like number of epochs, which are not possible to include in this code.</br></br>

The idea here is to keep a set of possible values for each of the hyper parameters in arrays and execute each combination for 1, 2 or 3 epochs to get the best performing combination. After finding a combination, you might also want to further improve the performance of the model, by running the code again with a more refined set of hyper parameters.</br></br>

The code repository contains two Python scripts that demonstrate how to find the best set of hyper parameters.</br>
1. 'find_best_hyperparameters_dense_model.py' has a Dense Layer model with simpler data (28, 28, 1) and runs faster.
2. 'find_best_hyperparameters_cnn.py' is using a CNN to classify cifar10 data (32, 32, 3). This takes longer time to give the results.

### Dependencies
1. numpy
1. matplotlib
1. tensorflow

### Instructions
1. Clone this repository to your local machine.
2. Enable virtual environment by running scripts\activate
3. Install the required dependencies.
4. Run the Python scripts. Refer Steps for further details.

### Steps
1. Provide a set of possible values for the hyper parameters Learning Rate, Number of Hidden Layers and Nodes, Mini Batch Size, L2 Regularization and Dropout. Keep number of epochs as 1 or 2. Then run 'find_best_params' function to find the best combination.<b> Note that Step 1 will take more time to complete if the number of combinations are more.</b>. To reduce execution time, the better approach is to skip regularization check in Step 1 by setting 'l2_factors' and 'dropout_rates' to [0]. Then use the best 1 or 2 results of Step 1 in Step 2 to find good regularizations (using more epochs).
2. Finds if applying regularization improves performance by trying different combinations of L2 and Dropout. This step is done on more number of epochs.
3. Train model using the best configuration identified from Step 1 and Step 2.

### License
This project is licensed under the MIT License.