import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data (replace with your own)
texts = [
    "This is a positive review.",
    "This movie is great!",
    "I didn't like this book.",
    "The acting was terrible.",
    "This movie is amazing!",
    "I didn't enjoy this film.",
    "The plot was captivating."
]
labels = np.array([1, 1, 0, 0, 1, 0, 1])  # 1 for positive, 0 for negative

# Tokenize the text data
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to ensure consistent length
maxlen = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=maxlen)

# Load pre-trained word embeddings (e.g., GloVe)
embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create embedding matrix
word_index = tokenizer.word_index
embedding_dim = 100
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Define the model
model = Sequential()
model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, labels, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(padded_sequences, labels)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Classify a set of new reviews
texts_to_classify = [
    "The movie is really great",
    "The movie is boring and terrible. I didn't like it",
    "The movie was rated really great. But I found it boring and terrible",
    "The movie was captivating"
]
new_sequences = tokenizer.texts_to_sequences(texts_to_classify)
new_padded_sequences = pad_sequences(new_sequences, maxlen=maxlen)
predictions = model.predict(new_padded_sequences)
for text, pred in zip(texts_to_classify, predictions):
    sentiment = "positive" if pred > 0.5 else "negative"
    print(f"Review: {text}, Sentiment: {sentiment}, Score: {pred[0]}")