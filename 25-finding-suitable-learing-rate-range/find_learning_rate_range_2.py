import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
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
        
        history = model.fit(train_images, train_labels, epochs=ep, batch_size=64,
                            validation_data=(test_images, test_labels))
        
        val_accs.append(min(history.history['val_accuracy']))
        val_losses.append(min(history.history['val_loss']))

    plt.figure(figsize=(12, 5))

    # Validation accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(lrs, val_accs, marker='o')
    plt.xscale('log')
    plt.title('Validation Accuracy')
    plt.xlabel('Learning Rate')
    plt.ylabel('Min Accuracy')

    # Validation loss plot
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
    
    history = model.fit(train_images, train_labels, epochs=ep, batch_size=64,
                        validation_data=(test_images, test_labels))
    
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

# Run below section to get a suitable learning rate range.
# In most cases, 1-2 epochs for each learning rate will give you a clear indication.
# If needed, use a shorter range/more number of epochs and run the test again to find a better learning rate.
lrs = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 7e-1, 1.0]
epochs = 2
find_lr_range(lrs, epochs)

# After finding the best learning rate(best_lr) by running above section,
# train model using new learning rate by running the below code.
'''lr = 0.001
epochs = 10
train_model(lr, epochs)'''
