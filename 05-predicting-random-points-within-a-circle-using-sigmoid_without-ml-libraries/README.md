Points within a Circle Prediction Neural Network
In a previous project (02-perceptron-predict-within-a-circle), I developed a neural network to predict whether a randomly selected point falls within a circle. The initial implementation utilized a perceptron with a step activation function. You might be aware of the limitations of step activation, which is effective only for linear separation. So to make prediction, the idea was to split the circle to a set of linear planes each doing a prediction and to combine all those to do the final prediction.

Model Enhancement
The current model employs a sigmoid activation function with a neural network structure of layers [2, 30, 2]. Input features include x and y coordinates, and the output classifies points as inside or outside the circle. This modification allows for a more nuanced representation, overcoming the linear separation constraint.
Here I am not using any Neural network libraries. The SGD and the back propogation logic is implemented in the code itself. Explanations are not provided here. If you need more details on this code please refer '04-recognize-digits-using-sigmoid-perceptron_without-ml-libraries' which uses the same functions and has the detailed explanations inline.

Code Features
In addition to the upgraded model architecture, the code includes functionalities to generate training and test points. Post-training, the script visualizes the test results on the final epoch, leveraging the learned network weights and biases. Also there is a second python file which does the same prediction for a random polygon space. Do check that also.

Installation of matplotlib is required

Feel free to explore the code and experiment with the parameters for a deeper understanding of the model's performance.