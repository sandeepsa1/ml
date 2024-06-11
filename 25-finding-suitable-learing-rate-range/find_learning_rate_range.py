import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

def find_lr_range(lrs, ep):
    val_accs = []
    val_losses = []

    for lr in lrs:
        print(f"Training with learning rate: {lr}")
        model = create_model()
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                    loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        history = model.fit(x_train, y_train, epochs=2, batch_size=64, 
                            validation_data=(x_test, y_test), verbose=0)
        
        val_accs.append(min(history.history['val_accuracy']))
        val_losses.append(min(history.history['val_loss']))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(lrs, val_accs, marker='o')
    plt.xscale('log')
    plt.title('Validation Accuracy')
    plt.xlabel('Learning Rate')
    plt.ylabel('Min Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(lrs, val_losses, marker='o')
    plt.xscale('log')
    plt.title('Validation Loss')
    plt.xlabel('Learning Rate')
    plt.ylabel('Min Loss')

    plt.show()
    
    best_lr = lrs[np.argmin(val_losses)]
    print(f"Best learning rate: {best_lr}")
    return best_lr

def train_model(lr, ep):
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, epochs=ep, batch_size=64,
                        validation_data=(x_test, y_test))
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.show()

# Run below section to get a suitable learning rate.
# In most cases, 1-2 epochs for each learning rate will give a clear indication.
# If needed, use a shorter range/more number of epochs and run the test again to find a better learning rate.
lrs = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 7e-1, 1.0]
epochs = 2
best_lr = find_lr_range(lrs, epochs)
# Second run with a shorter range. Here above run provided value 0.05. Run this after running above section.
'''lrs = [0.042, 0.044, 0.046, 0.048, 0.05, 0.052, 0.054, 0.056, 0.058, 0.059]
epochs = 2
best_lr = find_lr_range(lrs, epochs)'''


# After finding the best learning rate(best_lr) by running above section,
# train model using new learning rate by running the below code.
'''lr = 0.05
epochs = 10
train_model(lr, epochs)'''
