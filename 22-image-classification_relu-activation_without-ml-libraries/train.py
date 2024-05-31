import vectorize_images as im
import nn_2layers as nn

def run_training():
    train_len = 1000
    train_path = "data/cats_and_dogs/training_set/"
    (X_train, Y_train) = im.generate_datavectors(train_len, train_path)
    (X_train, Y_train) = im.shuffle_data(X_train, Y_train)
    # Use this function to check if mages properly shuffled. Change train_len to 10 to test this.
    # im.plotimageswithlabel(X_train, Y_train)

    # Generate [12288, 400] matrix. For 400 train samples. Each column in matrix is each sample.
    # print(X_train.shape) # (400, 64, 64, 3)
    X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
    # print(X_train_flatten.shape) # (12288, 400) # 12288 = 64 * 64 * 3
    # print(X_train_flatten[0])

    y_layer = 1
    # Change below hyper parameters to compare performances
    h_layer = 100
    learning_rate = 0.014
    epochs = 5000
    (W1, b1, W2, b2) = nn.iterate(X_train_flatten, Y_train, h_layer, y_layer, learning_rate, epochs)
    nn.save_weights_and_biases(W1, b1, W2, b2)

run_training()