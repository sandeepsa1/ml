import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

# To reduce variance, number of training samples can be increased.
# Another ways is to apply regularization.
def generate_data(n_samples):
    return make_regression(n_samples=n_samples, n_features=1, noise=20)

# Generate smaller and larger datasets
X_small, y_small = generate_data(200)
X_large, y_large = generate_data(1000)

X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(X_small, y_small, test_size=0.2, random_state=42)
X_train_large, X_test_large, y_train_large, y_test_large = train_test_split(X_large, y_large, test_size=0.2, random_state=42)

def build_and_train_model(X_train, y_train, layers, epochs, regularization=None):
    model = Sequential()
    model.add(Dense(10, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=regularization))
    for _ in range(layers - 1):
        model.add(Dense(10, activation='relu', kernel_regularizer=regularization))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, verbose=0, validation_split=0.2)
    return model, history

# Train model on small dataset without regularization
model_small_no_reg, history_small_no_reg = build_and_train_model(X_train_small, y_train_small, layers=3, epochs=100)

# Train model on large dataset with regularization
regularizer = l2(0.01)
model_large_reg, history_large_reg = build_and_train_model(X_train_large, y_train_large, layers=3, epochs=100, regularization=regularizer)

plt.figure(figsize=(12, 6))
plt.plot(history_small_no_reg.history['loss'], label='Training Loss (Small Dataset, No Regularization)')
plt.plot(history_small_no_reg.history['val_loss'], label='Validation Loss (Small Dataset, No Regularization)')
plt.plot(history_large_reg.history['loss'], label='Training Loss (Large Dataset, Regularization)')
plt.plot(history_large_reg.history['val_loss'], label='Validation Loss (Large Dataset, Regularization)')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

y_pred_train_small_no_reg = model_small_no_reg.predict(X_train_small)
y_pred_test_small_no_reg = model_small_no_reg.predict(X_test_small)
y_pred_train_large_reg = model_large_reg.predict(X_train_large)
y_pred_test_large_reg = model_large_reg.predict(X_test_large)

r2_small_no_reg_train = r2_score(y_train_small, y_pred_train_small_no_reg)
r2_small_no_reg_test = r2_score(y_test_small, y_pred_test_small_no_reg)
r2_large_reg_train = r2_score(y_train_large, y_pred_train_large_reg)
r2_large_reg_test = r2_score(y_test_large, y_pred_test_large_reg)

mae_small_no_reg_train = mean_absolute_error(y_train_small, y_pred_train_small_no_reg)
mae_small_no_reg_test = mean_absolute_error(y_test_small, y_pred_test_small_no_reg)
mae_large_reg_train = mean_absolute_error(y_train_large, y_pred_train_large_reg)
mae_large_reg_test = mean_absolute_error(y_test_large, y_pred_test_large_reg)

range_y_small = np.ptp(y_small)
range_y_large = np.ptp(y_large)

percentage_error_small_no_reg_train = (mae_small_no_reg_train / range_y_small) * 100
percentage_error_small_no_reg_test = (mae_small_no_reg_test / range_y_small) * 100
percentage_error_large_reg_train = (mae_large_reg_train / range_y_large) * 100
percentage_error_large_reg_test = (mae_large_reg_test / range_y_large) * 100

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_small, y_small, color='blue', alpha=0.5, label='Data')
plt.plot(X_train_small, y_pred_train_small_no_reg, color='red', label='Predicted (Small Dataset, No Regularization)')
plt.title(f'Data and Predictions (Small Dataset, No Regularization)\nTrain R^2: {r2_small_no_reg_train:.2f}, Test R^2: {r2_small_no_reg_test:.2f}\nTrain % Error: {percentage_error_small_no_reg_train:.2f}%, Test % Error: {percentage_error_small_no_reg_test:.2f}%')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_large, y_large, color='blue', alpha=0.5, label='Data')
plt.plot(X_train_large, y_pred_train_large_reg, color='red', label='Predicted (Large Dataset, Regularization)')
plt.title(f'Data and Predictions (Large Dataset, Regularization)\nTrain R^2: {r2_large_reg_train:.2f}, Test R^2: {r2_large_reg_test:.2f}\nTrain % Error: {percentage_error_large_reg_train:.2f}%, Test % Error: {percentage_error_large_reg_test:.2f}%')
plt.legend()

plt.show()
