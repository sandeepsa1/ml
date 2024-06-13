import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

def create_model(nodes):
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28))]) # Input
    for n in nodes: # Hidden layers
        model.add(tf.keras.layers.Dense(n, activation='relu'))    
    model.add(tf.keras.layers.Dense(10, activation='softmax')) # Output

    return model

def fit_model(model, lr, ep):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])        
    history = model.fit(x_train, y_train, epochs=ep, batch_size=64, 
                        validation_data=(x_test, y_test), verbose=0)
    
    return history

def find_best_node_config(hidden_layer_nodes, ep, lr):
    val_accs = []
    val_losses = []
    
    for nodes in hidden_layer_nodes:
        print(f"Training using {len(nodes)} hidden layers with nodes size: {nodes}")
        m = create_model(nodes)
        history = fit_model(m, lr, ep)
        
        val_accs.append(min(history.history['val_accuracy']))
        val_losses.append(min(history.history['val_loss']))
        
    node_labels = [','.join(map(str, nodes)) for nodes in hidden_layer_nodes]
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(node_labels, val_accs)
    plt.title('Validation Accuracy')
    plt.xlabel('Node Configurations')
    plt.ylabel('Min Accuracy')
    
    # Hide some x labels.
    plt.xticks(ticks=range(len(node_labels)), labels=[label if i % 2 == 0 else '' for i, label in enumerate(node_labels)])
    # Show best accuracy in the plot.
    best_index1 = val_accs.index(max(val_accs))
    plt.annotate(f'Best layer-node config: ({node_labels[best_index1]})\nAccuracy: {round(val_accs[best_index1], 3)}',
                 (node_labels[best_index1], val_accs[best_index1]), textcoords="offset points", xytext=(0,-15), ha='center', color='brown')

    plt.subplot(2, 1, 2)
    plt.plot(node_labels, val_losses, marker='o')
    plt.title('Validation Loss')
    plt.xlabel('Node Configurations')
    plt.ylabel('Min Loss')
    
    plt.xticks(ticks=range(len(node_labels)), labels=[label if i % 2 == 0 else '' for i, label in enumerate(node_labels)])
    # Show best loss in the plot.
    best_index2 = val_losses.index(min(val_losses))
    plt.annotate(f'Best layer-node config: ({node_labels[best_index2]})\nLoss: {round(val_losses[best_index2], 3)}',
                 (node_labels[best_index2], val_losses[best_index2]), textcoords="offset points", xytext=(0,15), ha='center', color='brown')

    plt.subplots_adjust(hspace=0.5)
    plt.show()

    best_node_config = hidden_layer_nodes[np.argmin(val_losses)]
    best_acc = val_accs[np.argmax(val_accs)]
    print(f"Best node configuration: {best_node_config}")
    print(f"Best accuracy: {best_acc}")
    return best_node_config

def train_model(best_node_config, ep, lr):
    m = create_model(best_node_config)
    history = fit_model(m, lr, ep)
    
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


lr = 0.001 # This is fixed. To tune learning rate, try '25-finding-best-learning-rate-range' code.

# Run below section to get the best configuration of number of layers and nodes.
# May have to run few more epochs than learning rates to get the best results. So go for 3-5 epochs minimum.
# May also need other runs with different combinations than given below, since there is no range possible and we are trying random number of layers and nodes.
# Also note that in rare cases, for accuracy and losses, different layer-node configurations
# may provide better results. If so, try both in the next section and select the one with less variance.
epochs = 5
hidden_layer_nodes = [ [4, 6], [7, 12], [12, 12], [20, 30], [50, 70],
            [7, 12, 8], [10, 10, 10], [19, 15, 12], [25, 35, 40], [40, 50, 55],
            [5, 4, 4, 3], [8, 10, 10, 18], [16, 8, 4, 11], [19, 25, 30, 35], [40, 45, 50, 55] ]
best_node_config = find_best_node_config(hidden_layer_nodes, epochs, lr)

# After finding the best configuration of number of layers and nodes within each layer (best_node_config),
# train model using this configuration by running the below code.
'''epochs = 40
best_node_config = [19, 15, 12] # Provide best result here.
train_model(best_node_config, epochs, lr)'''