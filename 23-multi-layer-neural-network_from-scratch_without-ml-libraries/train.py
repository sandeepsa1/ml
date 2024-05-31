import vectorize_images as im
import nn_multi_layers as nn

def run_training():
    train_len = 2000
    train_path = "data/cats_and_dogs/training_set/"
    (X_train, Y_train) = im.generate_datavectors(train_len, train_path)
    (X_train, Y_train) = im.shuffle_data(X_train, Y_train)
    # Use this function to check if images properly shuffled. Change train_len to 10 to test this.
    # im.plotimageswithlabel(X_train, Y_train)

    # Generate [12288, 400] matrix. For 400 train samples. Each column in matrix is each sample.
    # print(X_train.shape) # (400, 64, 64, 3)
    X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
    # print(X_train_flatten.shape) # (12288, 400) # 12288 = 64 * 64 * 3

    # Change below hyper parameters to compare performances
    layer_nodes = [10, 10, 1]
    learning_rate = 0.1
    epochs = 15000
    weights_and_biases = nn.iterate(X_train_flatten, Y_train, layer_nodes, learning_rate, epochs)
    
    nn.save_weights_and_biases(weights_and_biases)

run_training()