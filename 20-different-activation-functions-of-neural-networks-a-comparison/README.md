## Neural Network Activation Functions Comparison
Activation functions are a critical component of neural networks, determining the output of each neuron and introducing non-linearity into the model. This repository contains code to compare the performance of different activation functions in a neural network model using TensorFlow.


### Commonly used activation functions and their applications
1.  Sigmoid Function
    1. <b>Formula:</b> $ \[\sigma(x) = \frac{1}{1 + e^{-x}}\] $
    1. <b>Range:</b> (0, 1)
    1. <b>Usage:</b>
        1. Commonly used in the output layer for binary classification problems.
        1. Historically used in hidden layers, but less common now due to issues like vanishing gradients.
    1. <b>Pros:</b>
        1. Smooth gradient.
        1. Output values bound between 0 and 1.
    1. <b>Cons:</b>
        1. Can cause vanishing gradient problem.
        1. Outputs not zero-centered, which can slow down convergence.

2.  Hyperbolic Tangent (Tanh) Function
    1. <b>Formula:</b> $ \[\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}\] $
    1. <b>Range:</b> (-1, 1)
    1. <b>Usage:</b>
        1. Often used in hidden layers of neural networks.
        1. Preferred over sigmoid for hidden layers because it outputs zero-centered values.
    1. <b>Pros:</b>
        1. Zero-centered outputs, making optimization easier.
    1. <b>Cons:</b>
        1. Can still cause vanishing gradient problem, though less severe than sigmoid.

3.  Rectified Linear Unit (ReLU) Function
    1. <b>Formula:</b> $ \[\text{ReLU}(x) = \max(0, x)\] $
    1. <b>Range:</b> (-1, ∞)
    1. <b>Usage:</b>
        1. Most widely used activation function in hidden layers.
        1. Suitable for almost all types of neural networks.
    1. <b>Pros:</b>
        1. Efficient computation.
        1. Reduces likelihood of vanishing gradients.
    1. <b>Cons:</b>
        1. Can cause "dead neurons" (neurons that output zero and stop learning).

4.  Leaky ReLU Function
    1. <b>Formula:</b> $ \[\text{Leaky ReLU}(x) = \begin{cases}    x & \text{if } x > 0 \    \\alpha x & \text{otherwise} \end{cases}\] $
    1. <b>Range:</b> (-∞, ∞)
    1. <b>Usage:</b>
        1. Used to mitigate the "dying ReLU" problem by allowing a small gradient when the unit is not active.
    1. <b>Pros:</b>
        1. Prevents dead neurons.
    1. <b>Cons:</b>
        1. The slope of 𝛼 needs to be determined and can affect performance.

5.  Parametric ReLU (PReLU) Function
    1. <b>Formula:</b> $\[
\text{PReLU}(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0
\end{cases}
\]$
    1. <b>Range:</b> (-∞, ∞)
    1. <b>Usage:</b>
        1. Similar to Leaky ReLU, but 𝛼 is learned during training.
    1. <b>Pros:</b>
        1. Adaptable and can lead to better performance.
    1. <b>Cons:</b>
        1. Adds complexity as 𝛼 is a parameter to be learned.

6.  Exponential Linear Unit (ELU) Function
    1. <b>Formula:</b> $$\[
\text{ELU}(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha (e^x - 1) & \text{if } x \leq 0
\end{cases}
\]$
    1. <b>Range:</b> (-𝛼, ∞)
    1. <b>Usage:</b>
        1. Used in hidden layers.
    1. <b>Pros:</b>
        1. Helps to mitigate vanishing gradient.
        1. Output is zero-centered.
    1. <b>Cons:</b>
        1. More computationally expensive than ReLU.

7.  Softmax Function
    1. <b>Formula:</b> $\[
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
\]$
    1. <b>Range:</b> (0, 1) for each class, and the sum of outputs is 1.
    1. <b>Usage:</b>
        1. Used in the output layer of neural networks for multi-class classification.
    1. <b>Pros:</b>
        1. Provides probability distribution across multiple classes.
    1. <b>Cons:</b>
        1. Not suitable for hidden layers.


### Choosing the Right Activation Function
1.  Output Layer:
    1. Binary Classification: Sigmoid.
    1. Multi-class Classification: Softmax.
2.  Hidden Layers:
    1. ReLU is generally a good default choice.
    2. Leaky ReLU or PReLU can be used to address dying ReLU issues.
    3. ELU can be used if zero-centered output is beneficial and computational cost is not a concern.




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
1. sigmoid
1. tanh
1. relu
1. leaky_relu
1. elu

### Results
The training and validation accuracy and loss for each activation function are plotted over epochs in separate subplots for comparison.

#### Dependencies
1. Python 3.x
1. Matplotlib
1. Tensorflow