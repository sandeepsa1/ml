import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

loss_functions = {
    'Binary Crossentropy': 'binary_crossentropy',
    'Hinge': 'hinge',
    'Squared Hinge': 'squared_hinge',
    'Mean Squared Error': 'mean_squared_error',
    'Huber': tf.losses.Huber(delta=1.0)
}

history = {}
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
for name, loss_function in loss_functions.items():
    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
    hist = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=0)
    history[name] = hist
    plt.plot(hist.history['loss'], label=name)
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
for name, loss_function in loss_functions.items():
    plt.plot(history[name].history['val_loss'], label=name)
# plt.legend(loc='upper right')

plt.show()
