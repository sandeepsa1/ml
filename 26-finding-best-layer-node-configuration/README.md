## Finding a suitable configuration of number of hidden layers and nodes within each layer
Finding the best configuration of number of hidden layers and nodes within each of these layers is crucial for effectively training neural networks.</br>
This repository contains two Python scripts that demonstrate how to find the best configuration of number of hidden layers and nodes.
The 2 scripts provided here follow the same approach. The first one uses simple data (28, 28, 1) and a dense layer model, which gives the result faster. Second script is using a CNN to classify cifar10 data (32, 32, 3) and takes longer time to give the results.

### Dependencies
1. numpy
1. matplotlib
1. tensorflow

### Instructions
1. Clone this repository to your local machine.
2. Enable virtual environment by running scripts\activate
3. Install the required dependencies.
4. Run the Python scripts as mentioned below.
   1. First run the find_best_node_config method by providing a set of random hidden layers and number of nodes for each layer, as arrays.  Upto 5 epochs for each configuration may be needed to get a clear indication.
   1. Run the above method again with a different set of configurations if required, to find the best layer-node configuration.
   1. Then run the train_model method using the best layer-node configuration.

### License
This project is licensed under the MIT License.