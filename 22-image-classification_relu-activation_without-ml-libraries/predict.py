import vectorize_images as im
import nn_2layers as nn
import numpy as np

test_len = 500
test_path = "data/cats_and_dogs/test_set/"
(X_test, Y_test) = im.generate_datavectors(test_len, test_path)
(X_test, Y_test) = im.shuffle_data(X_test, Y_test)

# Load the saved weight and bais data after training.
loaded_data = np.load('model_parameters.npz')
W1 = loaded_data['W1']
b1 = loaded_data['b1']
W2 = loaded_data['W2']
b2 = loaded_data['b2']
# print(W2.shape)

X_test_flatten = X_test.reshape(X_test.shape[0], -1).T
(Z1, A1, Z2, A2) = nn.forward(X_test_flatten, W1, b1, W2, b2)
# print(A2.shape)
predictions = A2 > 0.5
# print(predictions.shape)

correct_predictions = np.sum(predictions == Y_test)
# print(len(correct_predictions))

print ('Accuracy Percentage: ' + str((correct_predictions / predictions.shape[1]) * 100))
