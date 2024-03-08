import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import random

#------------------------------Data Load------------------------------
def is_inside_circle(x, y, radius):
    return 1 if(x**2 + y**2 <= radius**2) else 0

def vectorize(j):
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e

training_points, test_points = 10000, 500
tr_pts = np.random.uniform(-2, 2, (training_points, 2))
tst_pts = np.random.uniform(-2, 2, (test_points, 2))
training, test = [], []

for point in tr_pts:
    x, y = point
    training.append([np.array([[x], [y]]), vectorize([is_inside_circle(x, y, 1)])])
for point in tst_pts:
    x, y = point
    test.append([np.array([[x], [y]]), is_inside_circle(x, y, 1)])
#------------------------------Data Load------------------------------

#------------Sigmoid activation function and its derivative-----------
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoidDerivative(x):
    return sigmoid(x)*(1-sigmoid(x))
#------------Sigmoid activation function and its derivative-----------

#-----Check performance after each epoch and plot after final run-----
def findActivation(a):
    for b, w in zip(biases, weights):
        a = sigmoid(np.dot(w, a)+b)
    return a    

def checkPerformance(i):
    s = 0
    points = []
    res = [(x, y, np.argmax(findActivation(x))) for (x, y) in test]
    for x, actual, calculated in res:
        if(int(actual == calculated)):
            s += 1
        if(i == (epochs-1)):
            color = "red" if(calculated == 1) else "blue"
            points.append([x[0], x[1], color])
    
    print ("Epoch {0}: {1} / {2}".format(i+1, s, len(test)))
    if(i == (epochs-1)):
        plot_test_points(points)

def plot_test_points(points):
    fig, ax = plt.subplots()
    circle = plt.Circle((0, 0), 1, edgecolor='black', facecolor='none')
    ax.add_patch(circle)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xticks([])
    ax.set_yticks([])

    for x, y, color in points:
        plt.scatter(x, y, c=color, alpha=0.5)
    plt.show()
#-----Check performance after each epoch and plot after final run-----

#-----------------Initialize layers, weights and bias-----------------
layers = [2, 30, 2]
biases = []
weights = []
for i in range(len(layers)):
    if(i > 0):
        biases.append(np.random.randn(layers[i], 1))
    if(i < len(layers)-1):
        weights.append(np.random.randn(layers[i+1], layers[i]))
#-----------------Initialize layers, weights and bias-----------------

#-----------------------Implement SGD algorithm-----------------------
learningRate = 2.5
epochs = 10
batchSize = 10

for i in range(epochs):
    random.shuffle(training)
    batches = [training[i:i + batchSize] for i in range(0, len(training), batchSize)]
    
    for batch in batches:
        biasGradients = [np.zeros(b.shape) for b in biases]
        weightGradients = [np.zeros(w.shape) for w in weights]

        for x, y in batch:
            activation = x
            activations = [x]
            zs = []
            for b, w in zip(biases, weights):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = sigmoid(z)
                activations.append(activation)
                
            gradient = (activations[-1] - y) * sigmoidDerivative(zs[-1]) # Formula 1
            
            biasGradients[-1] = gradient
            weightGradients[-1] = np.dot(gradient, activations[-2].transpose())
            
            for layer in range(2, len(layers)):
                z = zs[-layer]
                sp = sigmoidDerivative(z)
                gradient = np.dot(weights[-layer+1].transpose(), gradient) * sp
                
                biasGradients[-layer] = gradient
                weightGradients[-layer] = np.dot(gradient, activations[-layer-1].transpose())

        weights = [w-(learningRate/len(batch))*grw for w, grw in zip(weights, weightGradients)] # Formula 3
        biases = [b-(learningRate/len(batch))*grb for b, grb in zip(biases, biasGradients)] # Formula 3

    checkPerformance(i)
#-----------------------Implement SGD algorithm-----------------------