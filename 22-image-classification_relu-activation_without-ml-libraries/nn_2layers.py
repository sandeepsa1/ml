import numpy as np

# Activation functions and its derivatives
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu_prime(x):
    return np.where(x > 0, 1, 0)

def sigmoid_prime(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

# Set random weights and biases
def set_weights_and_bias(x_layer, h_layer, y_layer):
    W1 = np.random.randn(h_layer, x_layer) * 0.01
    b1 = np.zeros((h_layer, 1))
    W2 = np.random.randn(y_layer, h_layer) * 0.01
    b2 = np.zeros((y_layer, 1))

    return (W1, b1, W2, b2)

# Forward propagation
def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    return (Z1, A1, Z2, A2)

# Cost function
def find_cost(A, Y, m):
    logval = np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A))
    return -1 * np.sum(logval) / m

# Backward propagation
def backward(Z1, A1, A2, W2, X, Y, m):
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis = 1, keepdims = True) / m

    dZ1 = np.dot(W2.T, dZ2) * relu_prime(Z1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis = 1, keepdims = True) / m

    return (dZ2, dW2, db2, dZ1, dW1, db1)

def update_weights_and_biases(W1, W2, b1, b2, dW1, dW2, db1, db2, lr):
    W1 = W1 - (lr * dW1)
    b1 = b1 - (lr * db1)
    W2 = W2 - (lr * dW2)
    b2 = b2 - (lr * db2)

    return (W1, b1, W2, b2)

def iterate(X_train_flatten, Y_train, h_layer, y_layer, learning_rate, epochs):
    x_layer = X_train_flatten.shape[0]
    (W1, b1, W2, b2) = set_weights_and_bias(x_layer, h_layer, y_layer)
    m = X_train_flatten.shape[1]

    for i in range(0, epochs):
        (Z1, A1, Z2, A2) = forward(X_train_flatten, W1, b1, W2, b2)
        cost = find_cost(A2, Y_train, m)

        (dZ2, dW2, db2, dZ1, dW1, db1) = backward(Z1, A1, A2, W2, X_train_flatten, Y_train, m)
        (W1, b1, W2, b2) = update_weights_and_biases(W1, W2, b1, b2, dW1, dW2, db1, db2, learning_rate)

        if(i % 100 == 0):
            print("Cost after " + str(i+100) + " epochs: " + str(cost))
            #print(W1)

    return (W1, b1, W2, b2)

def save_weights_and_biases(W1, b1, W2, b2):
    np.savez('model_parameters.npz', W1 = W1, b1 = b1, W2 = W2, b2 = b2)
