import vectorize_images as im
import nn_multi_layers as nn
import numpy as np

test_len = 1000
test_path = "data/cats_and_dogs/test_set/"
(X_test, Y_test) = im.generate_datavectors(test_len, test_path)
(X_test, Y_test) = im.shuffle_data(X_test, Y_test)

# Load the saved weight and bais data after training.
loaded_data = np.load('model_parameters.npz')
layers_len = len(loaded_data) // 2
weights_and_biases = {}
for l in range(1, layers_len + 1):
    weights_and_biases['W' + str(l)] = loaded_data['W' + str(l)]
    weights_and_biases['b' + str(l)] = loaded_data['b' + str(l)]


X_test_flatten = X_test.reshape(X_test.shape[0], -1).T
(z_vals, activations) = nn.forward(X_test_flatten, weights_and_biases)

# print(activations['A' + str(layers_len)].shape)
predictions = activations['A' + str(layers_len)] > 0.5
# print(predictions)

correct_predictions = np.sum(predictions == Y_test)
# print(len(correct_predictions))

print ('Accuracy Percentage: ' + str((correct_predictions / predictions.shape[1]) * 100))
