from keras.datasets import imdb
from keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import keras
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

#print(train_data[4])
#print(train_labels[4])

train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

#model.summary()

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['acc'])
history = model.fit(train_data, train_labels, epochs=5, validation_split=0.2)

results = model.evaluate(test_data, test_labels)
print(results)


# Convert newly entered text to same format as training data and
# replace already indexed words in imdb with its index
word_index = imdb.get_word_index()
def encode_text(text):
    tokens = text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.pad_sequences([tokens], MAXLEN)[0]

def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1,250))
    pred[0] = encoded_text
    result = model.predict(pred) 
    sentiment = "positive" if result[0] > 0.5 else "negative"
    confidence = result[0] if result[0] > 0.5 else 1 - result[0]
    print("Text '" + text + "' is of " + sentiment + " sentiment. Confidence : " + str(confidence))

texts_to_classify = [
    "The movie is great. I loved story and acting",
    "The movie is boring and terrible. I will not be watching it again. One of the worst movies I have watched"
]
for text in texts_to_classify:
    predict(text)