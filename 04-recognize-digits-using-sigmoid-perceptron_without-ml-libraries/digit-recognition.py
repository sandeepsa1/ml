import pickle
import gzip
import numpy as np
import random

#------------------------------Data Load------------------------------
def loadData():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    u = pickle._Unpickler( f )
    u.encoding = 'latin1'
    training, validation, test = u.load()
    f.close()
    return (training, test)

def vectorized(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

"""Initially the format of training and test is as below
training = [2]
training[0] = 50000 - 784 length arrays for each pixel
training[1] = 50000 - corresponding digits

To enable shuffling data in each epoch in SGD function, change data to below format
training = [50000]
training[0][0] = 784 length arrays for each pixel
training[0][1] = corresponding digit
Also the corresponding digits converted to the matching output format (10 items.
Value is 1 for the position of the digit, rest are 0)"""
def reformatData(data, test):
    x = [np.reshape(d, (784, 1)) for d in data[0]]
    y = [y if test == 1 else vectorized(y) for y in data[1]]
    return list(zip(x, y))

training, test = loadData()
training = reformatData(training, 0)
test = reformatData(test, 1)
#------------------------------Data Load------------------------------

#------------Sigmoid activation function and its derivative-----------
def sigmoid(x): # Activation function
    return 1.0/(1.0+np.exp(-x))

def sigmoidDerivative(x): # Derivative of sigmoid function σ'(X) = σ(X)( 1 - σ(X) ). Used in back propogation formulas
    return sigmoid(x)*(1-sigmoid(x))
#------------Sigmoid activation function and its derivative-----------

#------------------Check performance after each epoch-----------------
def findActivation(a):
    for b, w in zip(biases, weights):
        a = sigmoid(np.dot(w, a)+b)
    return a    

def checkPerformance(): # Check how much improvement in learning happens after each epoch
    res = [(np.argmax(findActivation(x)), y) for (x, y) in test]
    s = sum(int(x == y) for (x, y) in res)
    print ("Epoch {0}: {1} / {2}".format(i+1, s, len(test)))
#------------------Check performance after each epoch-----------------

#-----------------Initialize layers, weights and bias-----------------
# In this sample, a 3 layer network is defined. Input 784, because image size is 28 * 28. Output 10, 1 for each digit.
# Any number of middle layers in between can be added with any number of nodes for each
layers = [784, 40, 10]
biases = []
weights = []
for i in range(len(layers)): # Initialize random biases and weights for each node ranging -1 to +1
    if(i > 0): # Bias of a perceptron starts from second layer till output layer
        biases.append(np.random.randn(layers[i], 1))
    if(i < len(layers)-1): # Weights start from first layer till end (excluding last)
        weights.append(np.random.randn(layers[i+1], layers[i]))
#-----------------Initialize layers, weights and bias-----------------

#-----------------------Implement SGD algorithm-----------------------
learningRate = 2.5
epochs = 20
batchSize = 10 # SGD works in batches

# Loop through each epoch
for i in range(epochs):
    random.shuffle(training) # Shuffle and split to mini batches for SGD
    batches = [training[i:i + batchSize] for i in range(0, len(training), batchSize)]
    
    """Step 1: Forward propogation.
    A simple perception works by summing up product of each input and weight + bias x1w1 + x2w2 + ... + xnwn + b.
    This sum is then applied to the activation function (here sigmoid) to get the output value. For example,
    consider first node of the second layer. It has 784 inputs and 784 weights. The summing up logic can beachieved by
    Numpy dot product np.dot. Then sigmoid function is applied to get the output value of this perceptron.
    This output is the input for the next layer. These calculations are done for all the layers till final output values.
    
    Step 2: Finding the gradient of the output layer
    Backpropogation algorithm works on 3 important formulas. To find gradient of output layer use first formula,
    δ = ∇C * σ'(z)    ∇C is the difference between final calculated output and the actual output.
                      σ'(z) is the sigmoid derivative of final layer (wx + b)
    
    Step 3: Back propogate from output to input layers to find the gradients of each layer
    Here we use second formula δ = ( (w of next layer)T dot(δ of next layer) ) * σ'(z of the layer)

    Step 4: Finaly update biases and weights for each layer according to the third formula
    Bias   b = b - (learningRate/batch size) * δ
    Weight w = w - (learningRate/batch size) * δ dot (activation of previous layer)T
    """    
    for batch in batches:
        # 0 based array similar to biases and weights to store gradients
        biasGradients = [np.zeros(b.shape) for b in biases]
        weightGradients = [np.zeros(w.shape) for w in weights]

        for x, y in batch:
            # Step 1: Forward propogation to find outputs. Activation a = σ(wx + b)
            activation = x
            activations = [x] # list to store all the activations, layer by layer
            zs = [] # list to store all the z vectors, layer by layer
            for b, w in zip(biases, weights):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = sigmoid(z)
                activations.append(activation)
                
            # Step 2: Finding the gradient of the output layer
            gradient = (activations[-1] - y) * sigmoidDerivative(zs[-1]) # Formula 1
            # Step 4
            biasGradients[-1] = gradient
            weightGradients[-1] = np.dot(gradient, activations[-2].transpose())
            
            #Step 3: Back propogate from output to input layers
            for layer in range(2, len(layers)):
                z = zs[-layer]
                sp = sigmoidDerivative(z)
                gradient = np.dot(weights[-layer+1].transpose(), gradient) * sp # Formula 2
                # Step 4
                biasGradients[-layer] = gradient
                weightGradients[-layer] = np.dot(gradient, activations[-layer-1].transpose())

        # Step 4: Update weights and biases using gradients calculated for each layer using back propogation
        weights = [w-(learningRate/len(batch))*grw for w, grw in zip(weights, weightGradients)] # Formula 3
        biases = [b-(learningRate/len(batch))*grb for b, grb in zip(biases, biasGradients)] # Formula 3

    checkPerformance()
#-----------------------Implement SGD algorithm-----------------------