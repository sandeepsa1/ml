import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import gc
from tensorflow.keras.backend import clear_session

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

def create_model(nodes, l2_factor = 0.0, dropout_rate = 0.0):
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28))]) # Input
    for n in nodes: # Hidden layers
        if (l2_factor > 0):
              model.add(tf.keras.layers.Dense(n, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
        else:
            model.add(tf.keras.layers.Dense(n, activation='relu'))  
        
        if (dropout_rate > 0):
            model.add(tf.keras.layers.Dropout(dropout_rate))
    
    if (l2_factor > 0): # Output
        model.add(tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l2_factor)))
    else:
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model

def fit_model(model, lr, batch, ep):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])        
    history = model.fit(x_train, y_train, epochs=ep, batch_size=batch, 
                        validation_data=(x_test, y_test), verbose=0)
    
    return history


def find_best_params(lrs, mini_batch_sizes, hidden_layer_nodes, l2_factors, dropout_rates, ep):
    val_accs = []
    val_losses = []
    node_labels = []
    
    for lr in lrs:
        for batch in mini_batch_sizes:
            for nodes in hidden_layer_nodes:
                 for l2 in l2_factors:
                    for drop in dropout_rates:
                        print(f"Learning Rate: {lr}. Batch Size: {batch}. Layer-node Config: {nodes}. L2: {l2}. Dropout: {drop}.")
                        m = create_model(nodes, l2, drop)
                        history = fit_model(m, lr, batch, ep)

                        clear_session()
                        gc.collect()

                        val_accs.append(min(history.history['val_accuracy']))
                        val_losses.append(min(history.history['val_loss']))

                        node_labels.append(f"{lr}, {batch}, [{','.join(map(str, nodes))}], {l2}, {drop}")
            
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(node_labels, val_accs)
    plt.title('Validation Accuracy')
    plt.xlabel('Hyper Parameters')
    plt.ylabel('Min Accuracy')
    
    # Hide some x labels.
    tick_size = (len(lrs) * len(mini_batch_sizes) * len(hidden_layer_nodes) * len(l2_factors) * len(dropout_rates)) // 5
    tick_size = 1 if tick_size < 1 else tick_size
    plt.xticks(ticks=range(len(node_labels)), labels=[label if i % tick_size == 0 else '' for i, label in enumerate(node_labels)])
    # Show best accuracy in the plot.
    best_index1 = val_accs.index(max(val_accs))
    plt.annotate(f'Best hyper parameters: {node_labels[best_index1]}\nAccuracy: {round(val_accs[best_index1], 3)}',
                 (node_labels[best_index1], val_accs[best_index1]), textcoords="offset points", xytext=(0,-15), ha='center', color='brown')

    plt.subplot(2, 1, 2)
    plt.plot(node_labels, val_losses, marker='o')
    plt.title('Validation Loss')
    plt.xlabel('Hyper Parameters')
    plt.ylabel('Min Loss')
    
    plt.xticks(ticks=range(len(node_labels)), labels=[label if i % tick_size == 0 else '' for i, label in enumerate(node_labels)])
    # Show best loss in the plot.
    best_index2 = val_losses.index(min(val_losses))
    plt.annotate(f'Best hyper parameters: ({node_labels[best_index2]})\nLoss: {round(val_losses[best_index2], 3)}',
                 (node_labels[best_index2], val_losses[best_index2]), textcoords="offset points", xytext=(0,15), ha='center', color='brown')

    plt.subplots_adjust(hspace=0.5)
    plt.show()

    best_hyper_parameters = node_labels[np.argmin(val_losses)]
    best_acc = val_accs[np.argmax(val_accs)]
    print(f"Best hyper Parameters: {best_hyper_parameters}")
    print(f"Best accuracy: {best_acc}")
    return best_hyper_parameters


def train_model(lr, batch, layer_node, ep):
    model = create_model(layer_node)
    history = fit_model(model, lr, batch, ep)
    
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


# Step 1: Find best hyper parameters including regularizations. Provide regularization values, if you want
# these to be checked for all combinations. But it is alomost always better to run Step 1
# without regularization (ie, Set l2_factors and dropout_rates to [0]).  Then use the best 1 or 2 results
# of Step 1 in Step 2 to find good regularizations (using more epochs) and this saves time as well.
# For below configuration, execution time in colab : 29 min
epochs = 2
lrs = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 7e-1, 1.0]
mini_batch_sizes = [64, 128]
hidden_layer_nodes = [ [12, 12], [20, 30], [50, 70], [10, 10, 10], [19, 15, 12], [25, 35, 40],
                      [40, 50, 55], [8, 10, 10, 18], [16, 8, 4, 11], [40, 45, 50, 55] ]
# l2_factors = [0, 1e-4, 1e-3]
# dropout_rates = [0, 0.2]
l2_factors = [0]
dropout_rates = [0]
best_params = find_best_params(lrs, mini_batch_sizes, hidden_layer_nodes, l2_factors, dropout_rates, epochs)

# Step 2: This step can be used to get faster results instead of trying regularizations on all models.
# Here we try different regularizations on the best 1 or 2 models from step 1 with more epochs.
# Set l2_factors and dropout_rates to [0] in Step 1 and run it before trying this.
# Execution time in colab : 22 min
'''epochs = 30
lrs = [0.01]
mini_batch_sizes = [64]
hidden_layer_nodes = [ [50, 70] ]
l2_factors = [0, 1e-4, 1e-3]
dropout_rates = [0, 0.1, 0.2, 0.4]
best_params = find_best_params(lrs, mini_batch_sizes, hidden_layer_nodes, l2_factors, dropout_rates, epochs)

epochs = 30
lrs = [0.05]
mini_batch_sizes = [128]
hidden_layer_nodes = [ [40, 50, 55] ]
l2_factors = [0, 1e-4, 1e-3]
dropout_rates = [0, 0.1, 0.2, 0.4]
best_params = find_best_params(lrs, mini_batch_sizes, hidden_layer_nodes, l2_factors, dropout_rates, epochs)'''

# Step 3: Train model using the best configuration.
# Execution time in colab : 3 min
# Results for below 2 configurations:
# Training Accuracy:    0.9791333079338074      0.9528833627700806
# Training Loss:        0.06383317708969116     0.17488744854927063
# Validation Accuracy:  0.9641000032424927      0.9318000078201294
# Validation Loss:      0.15174470841884613     0.25733834505081177
'''epochs = 50
lr = 0.01               # 0.05
batch = 64              # 128
layer_node = [50, 70]   # [40, 50, 55]
l2_factors = [0]
dropout_rates = [0.1]        # [0]
train_model(lr, batch, layer_node, epochs)'''