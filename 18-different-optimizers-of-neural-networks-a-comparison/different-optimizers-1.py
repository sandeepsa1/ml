import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta

np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1)

model = Sequential([
    Dense(1, input_shape=(1,))
])

optimizers = {
    'SGD': SGD(learning_rate=0.01),
    'Adam': Adam(learning_rate=0.01),
    'RMSprop': RMSprop(learning_rate=0.01),
    'Adagrad': Adagrad(learning_rate=0.01),
    'Adadelta': Adadelta(learning_rate=0.01)
}

histories = {}
for name, optimizer in optimizers.items():
    model.compile(optimizer=optimizer, loss='mse')
    history = model.fit(X, y, epochs=50, verbose=0)
    histories[name] = history.history['loss']

plt.figure(figsize=(10, 6))
for name, loss in histories.items():
    plt.plot(loss, label=name)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curves for Different Optimizers')
plt.legend()
plt.grid(True)
plt.show()