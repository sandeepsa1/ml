import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Create even/odd labels
y_train_even_odd = (y_train % 2 == 0).astype(int)
y_test_even_odd = (y_test % 2 == 0).astype(int)

# One-hot encode digit labels
y_train_digits = to_categorical(y_train, 10)
y_test_digits = to_categorical(y_test, 10)

# Input layer
input_layer = Input(shape=(28, 28, 1))

# Shared convolutional layers
x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)

# Output layer for digit classification
digit_output = Dense(10, activation='softmax', name='digit_output')(x)

# Output layer for even/odd classification
even_odd_output = Dense(1, activation='sigmoid', name='even_odd_output')(x)

model = Model(inputs=input_layer, outputs=[digit_output, even_odd_output])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss={'digit_output': 'categorical_crossentropy', 'even_odd_output': 'binary_crossentropy'},
              metrics={'digit_output': 'accuracy', 'even_odd_output': 'accuracy'})

history = model.fit(X_train, {'digit_output': y_train_digits, 'even_odd_output': y_train_even_odd},
                    validation_data=(X_test, {'digit_output': y_test_digits, 'even_odd_output': y_test_even_odd}),
                    epochs=3, batch_size=32)


evaluation = model.evaluate(X_test, {'digit_output': y_test_digits, 'even_odd_output': y_test_even_odd})
print(len(evaluation))
print(f"Test Loss: {evaluation[0]}")
print(f"Digit Classification Accuracy: {evaluation[1]}")
print(f"Even/Odd Classification Accuracy: {evaluation[2]}")

plt.plot(history.history['digit_output_accuracy'], label='digit_train_accuracy')
plt.plot(history.history['val_digit_output_accuracy'], label='digit_val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Digit Classification Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['even_odd_output_accuracy'], label='even_odd_train_accuracy')
plt.plot(history.history['val_even_odd_output_accuracy'], label='even_odd_val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Even/Odd Classification Accuracy')
plt.legend()
plt.show()
