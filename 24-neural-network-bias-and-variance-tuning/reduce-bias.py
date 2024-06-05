import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# To improve bias (reduce underfitting), number of layers or nodes within the layers can be increased.
# Another ways is to increase the number of iterations
X, y = make_regression(n_samples=1000, n_features=1, noise=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_and_train_model(X_train, y_train, layers, epochs):
    model = Sequential()
    model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
    for _ in range(layers - 1):
        model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, verbose=0, validation_split=0.2)
    return model, history

model_few, history_few = build_and_train_model(X_train, y_train, layers=1, epochs=50)
model_more, history_more = build_and_train_model(X_train, y_train, layers=3, epochs=300)
# model_more2, history_more2 = build_and_train_model(X_train, y_train, layers=7, epochs=100)

plt.plot(history_few.history['loss'], label='Training Loss (Few Layers/Iterations)', color='blue')
plt.plot(history_more.history['loss'], label='Training Loss (More Layers/Iterations)', color='red')
# plt.plot(history_more2.history['loss'], label='Training Loss (More Layers)', color='yellow')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the models and plot
y_pred_train_few = model_few.predict(X_train)
y_pred_test_few = model_few.predict(X_test)
y_pred_train_more = model_more.predict(X_train)
y_pred_test_more = model_more.predict(X_test)

# Calculate R^2 scores
r2_few_train = r2_score(y_train, y_pred_train_few)
r2_few_test = r2_score(y_test, y_pred_test_few)
r2_more_train = r2_score(y_train, y_pred_train_more)
r2_more_test = r2_score(y_test, y_pred_test_more)

mae_few_train = mean_absolute_error(y_train, y_pred_train_few)
mae_few_test = mean_absolute_error(y_test, y_pred_test_few)
mae_more_train = mean_absolute_error(y_train, y_pred_train_more)
mae_more_test = mean_absolute_error(y_test, y_pred_test_more)

range_y = np.ptp(y)

percentage_error_few_train = (mae_few_train / range_y) * 100
percentage_error_few_test = (mae_few_test / range_y) * 100
percentage_error_more_train = (mae_more_train / range_y) * 100
percentage_error_more_test = (mae_more_test / range_y) * 100

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', alpha=0.5, label='Data')
plt.plot(X_train, y_pred_train_few, color='red', label='Predicted (Few Layers/Iterations)')
plt.title(f'Data and Predictions (Few Layers/Iterations)\nTrain R^2: {r2_few_train:.2f}, Tes R^2: {r2_few_test:.2f}\nTrain % Error: {percentage_error_few_train:.2f}%, Test % Error: {percentage_error_few_test:.2f}%')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', alpha=0.5, label='Data')
plt.plot(X_train, y_pred_train_more, color='red', label='Predicted (More Layers/Iterations)')
plt.title(f'Data and Predictions (More Layers/Iterations)\nTrain R^2: {r2_more_train:.2f}, Test R^2: {r2_more_test:.2f}\nTrain % Error: {percentage_error_more_train:.2f}%, Test % Error: {percentage_error_more_test:.2f}%')
plt.legend()

plt.show()
