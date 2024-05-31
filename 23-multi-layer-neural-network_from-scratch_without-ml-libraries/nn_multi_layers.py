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

def set_weights_and_bias(layer_nodes):
    weights_and_biases = {}
    for l in range(1, len(layer_nodes)):
        weights_and_biases['W' + str(l)] = np.random.randn(layer_nodes[l], layer_nodes[l - 1]) * 0.01
        weights_and_biases['b' + str(l)] = np.zeros((layer_nodes[l], 1))
    
    return weights_and_biases

# Forward propagation
def forward(X, weights_and_biases):
    z_vals = {}
    activations = {}
    layers_len = len(weights_and_biases) // 2
    A = X
    for i in range(1, layers_len + 1):
        z = np.dot(weights_and_biases['W' + str(i)], A) + weights_and_biases['b' + str(i)]
        z_vals['Z' + str(i)] = z
        
        if(i == layers_len):
            A = sigmoid(z)
        else:
            A = relu(z)
        
        activations['A' + str(i)] = A    
    
    #print(z_vals['Z1'].shape, activations['A1']. shape)    
    return (z_vals, activations)

# Cost function
def find_cost(A, Y, m):
    logval = np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A))
    return -1 * np.sum(logval) / m

# Backward propagation
def backward(weights_and_biases, z_vals, activations, X, Y, m):
    layers_len = len(weights_and_biases) // 2
    gradients = {}

    dZ_last = activations['A' + str(layers_len)] - Y
    gradients['dZ' + str(layers_len)] = dZ_last
    gradients['dW' + str(layers_len)] = np.dot(dZ_last, activations['A' + str(layers_len - 1)].T) / m
    gradients['db' + str(layers_len)] = np.sum(dZ_last, axis = 1, keepdims = True) / m
    #print(gradients)
    
    for i in reversed(range(1, layers_len)):
        gradients['dZ' + str(i)] = np.dot(weights_and_biases['W' + str(i + 1)].T,
                                          gradients['dZ' + str(i + 1)]) * relu_prime(z_vals['Z' + str(i)])
        
        A = X if i == 1 else activations['A' + str(i - 1)]
        gradients['dW' + str(i)] = np.dot(gradients['dZ' + str(i)], A.T) / m
        gradients['db' + str(i)] = np.sum(gradients['dZ' + str(i)], axis = 1, keepdims = True) / m
    
    return gradients

def update_weights_and_biases(weights_and_biases, gradients, lr):
    layers_len = len(weights_and_biases) // 2
    for l in range(1, layers_len + 1):
        weights_and_biases['W' + str(l)] = weights_and_biases['W' + str(l)] - (lr * gradients['dW' + str(l)])
        weights_and_biases['b' + str(l)] = weights_and_biases['b' + str(l)] - (lr * gradients['db' + str(l)])
    
    return weights_and_biases

def iterate(X_train_flatten, Y_train, layer_nodes, learning_rate, epochs):
    x_len = X_train_flatten.shape[0]
    layer_nodes.insert(0, x_len) # Add input node lenth also
    #print(layer_nodes)
    
    weights_and_biases = set_weights_and_bias(layer_nodes)
    #print(params)
    #print(params['W1'].shape)
    m = X_train_flatten.shape[1]
    
    for i in range(0, epochs):
        (z_vals, activations) = forward(X_train_flatten, weights_and_biases)
        cost = find_cost(activations['A' + str(len(activations))], Y_train, m)

        gradients = backward(weights_and_biases, z_vals, activations, X_train_flatten, Y_train, m)

        weights_and_biases = update_weights_and_biases(weights_and_biases, gradients, learning_rate)

        if(i % 100 == 0):
            print("Cost after " + str(i+100) + " epochs: " + str(cost))
    
    return weights_and_biases

def save_weights_and_biases(weights_and_biases):
    layers_len = len(weights_and_biases) // 2
    params = {}
    
    for l in range(1, layers_len + 1):
        params[f'W{l}'] = weights_and_biases['W' + str(l)]
        params[f'b{l}'] = weights_and_biases['b' + str(l)]
    
    np.savez('model_parameters.npz', **params)