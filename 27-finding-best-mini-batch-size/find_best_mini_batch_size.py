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

def fit_model(mini_batch_size, lr, ep):    
    model = create_model()        
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])        
    history = model.fit(x_train, y_train, epochs=ep, batch_size=mini_batch_size, 
                        validation_data=(x_test, y_test), verbose=0)
    
    return history

def find_best_mini_batch_size(mini_batch_sizes, ep, lr):
    val_accs = []
    val_losses = []
    
    for mini_batch_size in mini_batch_sizes:
        print(f"Training with mini batch size: {mini_batch_size}")
        history = fit_model(mini_batch_size, lr, ep)
        
        val_accs.append(min(history.history['val_accuracy']))
        val_losses.append(min(history.history['val_loss']))
        
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(mini_batch_sizes, val_accs)
    plt.title('Validation Accuracy')
    plt.xlabel('Mini Batch Sizes')
    plt.ylabel('Min Accuracy')
    
    # Show best accuracy in the plot.
    best_index1 = val_accs.index(max(val_accs))
    plt.annotate(f'Best mini batch size: ({mini_batch_sizes[best_index1]})\nAccuracy: {round(val_accs[best_index1], 3)}',
                 (mini_batch_sizes[best_index1], val_accs[best_index1]), textcoords="offset points", xytext=(0,-15), ha='center', color='brown')

    plt.subplot(1, 2, 2)
    plt.plot(mini_batch_sizes, val_losses, marker='o')
    plt.title('Validation Loss')
    plt.xlabel('Mini Batch Sizes')
    plt.ylabel('Min Loss')
    
    # Show best loss in the plot.
    best_index2 = val_losses.index(min(val_losses))
    plt.annotate(f'Best mini batch size: ({mini_batch_sizes[best_index2]})\nLoss: {round(val_losses[best_index2], 3)}',
                 (mini_batch_sizes[best_index2], val_losses[best_index2]), textcoords="offset points", xytext=(0,15), ha='center', color='brown')

    plt.subplots_adjust(hspace=0.5)
    plt.show()

    best_mini_batch_size = mini_batch_sizes[np.argmin(val_losses)]
    best_acc = val_accs[np.argmax(val_accs)]
    print(f"Best mini batch size: {best_mini_batch_size}")
    print(f"Best accuracy: {best_acc}")
    return best_mini_batch_size

def train_model(mini_batch_size, ep, lr):
    history = fit_model(mini_batch_size, lr, ep)
    
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

    plt.subplots_adjust(hspace=0.5)
    plt.show()
    
    print(f"Training Accuracy: {history.history['accuracy'][-1]}")
    print(f"Training Loss: {history.history['loss'][-1]}")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1]}")
    print(f"Validation Loss: {history.history['val_loss'][-1]}")

# Run below section to get the mini batch size.
lr = 0.001
epochs = 5
mini_batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
best_mini_batch_size = find_best_mini_batch_size(mini_batch_sizes, epochs, lr)

# After finding the best mini batch size(best_mini_batch_size) by running above section,
# train model using new mini batch size by running the below code.
'''lr = 0.001
epochs = 10
mini_batch_size = 32
train_model(mini_batch_size, epochs, lr)'''