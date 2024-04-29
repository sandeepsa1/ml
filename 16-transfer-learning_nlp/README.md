The idea of Transfer Learning to reuse an already trained mode to do a different/similar task.
We use weights/architecture/training data/parameters of already trained network

1. sentiment_analysis.py
A simple example of transfer learning for natural language processing (NLP) using pre-trained word embeddings and a simple neural network classifier in TensorFlow/Keras. We are using pre-trained word embeddings as fixed weights in the embedding layer of our neural network model. and they are not re-trained during the training process. This allows us to benefit from the semantic information captured by the pre-trained word embeddings while training our model on a specific task with limited data.

How this works
    1. Loading pre-trained word embeddings:
    Load pre-trained word embeddings from a pre-trained word embeddings file (glove.6B.100d). These embeddings have been trained on a large corpus of text data.
    2. Creating an embedding matrix:
    Create an embedding matrix using the words in our vocabulary and the pre-trained word embeddings. Each row of the embedding matrix corresponds to a word in our vocabulary, and the values in each row are the pre-trained word embeddings for that word.
    3. Defining the model:
    Define a neural network model using TensorFlow/Keras. The first layer of the model is an embedding layer initialized with the embedding matrix we created earlier. This layer acts as a lookup table that maps each word index in our input sequences to its corresponding pre-trained word embedding. Since we set trainable=False, the weights of this embedding layer are not updated during training.
    4. Compiling and training the model:
    Compile the model using an appropriate loss function and optimizer and train it on our dataset of text samples and corresponding labels. During training, the weights of the LSTM layer and the Dense layer are updated based on the gradients of the loss function with respect to the model parameters.
    5. Evaluation:
    Evaluate the trained model on the test data to measure its performance in terms of loss and accuracy.
    6. Classification:
    Try to classify a set of reviews after the training. Note that our set of sample data is really small. So for classification, we should be including some of the words in sample data to get the correct results. We can do the training on actual IMDB dataset instead of small sample data, to get better results.


Instructions
Before running the code, download glove.6B.100d.txt file from the below url and keep it in the same folder of the code file.
    https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt
Ensure you have Python installed on your system.
Install the required libraries listed below.
Run the Python scripts to see how Transfer Learning works.

Requirements
numpy
tensorflow

License
This code is provided under the MIT License.