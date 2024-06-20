## Multitask Learning with MNIST
This project demonstrates a multitask learning approach using the MNIST dataset, where a single neural network is trained to perform two tasks simultaneously:
1. <b>Digit Classification</b>: Predict the digit (0-9).
2. <b>Even/Odd Classification</b>: Predict whether the digit is even or odd.

### Dependencies
1. numpy
2. matplotlib
3. tensorflow
4. sklearn

### Instructions
1. Clone this repository to your local machine.
2. Enable virtual environment by running scripts\activate
3. Install the required dependencies.
4. Run the Python script.

### Dataset
The MNIST dataset is used, which consists of 70,000 grayscale images of handwritten digits (0-9) with a resolution of 28x28 pixels. The dataset is split into 60,000 training images and 10,000 test images.

### Model Architecture
The model architecture includes:
1. Shared convolutional layers for feature extraction.
2. Two separate output layers:
   1. One for digit classification using softmax activation.
   1. One for even/odd classification using sigmoid activation.

### Custom Loss Function
A custom loss function is used to combine the loss for both tasks:
1. Categorical cross-entropy for digit classification.
2. Binary cross-entropy for even/odd classification.

### Training and Evaluation
The model is trained and evaluated on both tasks simultaneously. The training and validation accuracy for both tasks are plotted to visualize the training process.

### Summary
This project demonstrates how to build and train a multitask learning model using the MNIST dataset. The model predicts both the digit and whether the digit is even or odd, using the same model.