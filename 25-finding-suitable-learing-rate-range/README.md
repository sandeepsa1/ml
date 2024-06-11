## Finding a suitable learning rate range and thereby finding the best learning rate
Finding a suitable learning rate range is crucial for effectively training neural networks.</br>
This repository contains two Python scripts that demonstrate how to find the best learning rate range and from this range find the most suitable learning rate.
The 2 scripts provided here follow the same approach. The first one uses simple data and a simpler model, which gives the result faster. Second script is using a CNN to classify cifar10 data and takes longer time to give the results.

### Dependencies
1. numpy
1. matplotlib
1. tensorflow

### Instructions
1. Clone this repository to your local machine.
2. Enable virtual environment by running scripts\activate
3. Install the required dependencies.
4. Run the Python scripts as mentioned below.
   1. First run the find_lr_range method by providing a set of random learning rates.. One or two epochs for each learning rate should be enough to get a clear indication.
   1. Run the above method again with a shorter range of learning rates or more number of epochs if required, to find the best learning rate.
   1. Then run the train_model method using the best learning rate.

### License
This project is licensed under the MIT License.