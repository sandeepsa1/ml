import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

def create_model(activation_function):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512, activation=activation_function),
        Dense(512, activation=activation_function),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

activation_functions = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'softmax']

histories = {}
for activation_function in activation_functions:
    print(f'Training with {activation_function} activation function...')
    model = create_model(activation_function)
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=4, batch_size=128, verbose=1)
    histories[activation_function] = history

plt.figure(figsize=(14, 7))
plt.subplot(2, 2, 1)
for activation_function in activation_functions:
    plt.plot(histories[activation_function].history['accuracy'], label=f'{activation_function} - Train')
plt.title('Model Accuracy - Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(2, 2, 2)
for activation_function in activation_functions:
    plt.plot(histories[activation_function].history['val_accuracy'], '--', label=f'{activation_function} - Val')
plt.title('Model Accuracy - Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(2, 2, 3)
for activation_function in activation_functions:
    plt.plot(histories[activation_function].history['loss'], label=f'{activation_function} - Train')
plt.title('Model Loss - Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 4)
for activation_function in activation_functions:
    plt.plot(histories[activation_function].history['val_loss'], '--', label=f'{activation_function} - Val')
plt.title('Model Loss - Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()