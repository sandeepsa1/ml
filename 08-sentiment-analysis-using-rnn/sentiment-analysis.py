import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_features = 10000
maxlen = 500

def train_model():
    batch_size = 32
    epochs = 10

    # Load the data
    (x_train, y_train), (x_test, y_test) = imdb.load_data()

    # Preprocess the data
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    # Build the RNN model
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    
    print('Loss : ', loss)
    print('Accuracy : ', accuracy)
    print('Training Complete\n\n')

    return model

def classify_sentiment(model, text):
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])
    input_text = pad_sequences(sequences, maxlen=maxlen)

    prediction = model.predict(input_text)[0][0]

    sentiment = "positive" if prediction > 0.5 else "negative"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    print(f"Input text ' {text} ' is classified as {sentiment} with confidence {confidence:.2f}")


model = train_model()

texts_to_classify = [
    "The movie is great. I loved story and acting",
    "The movie is boring and terrible",
    "The movie was rated really great. But i found it boring and terrible",
    "The movie was rated really great. Many find it boring and terrible also"
]
for text in texts_to_classify:
    classify_sentiment(model, text)