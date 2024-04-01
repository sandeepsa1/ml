Simple Sentiment Analysis with Recurrent Neural Network (RNN)
As part of my NLP training, implementing a straightforward sentiment analysis tool using a simple Recurrent Neural Network (RNN). This tool aims to classify reviews from the IMDB database as either positive or negative.

Key Features:
sentiment-analysis
Model Architecture: The sentiment analysis model employs a basic RNN architecture.
Optimizer and Loss Function: Adam optimizer is utilized with binary_crossentropy loss function.
Dataset: IMDB database is used for training and evaluation.
Accuracy: This simple model achieves a respectable accuracy of 78%.

sentiment-analysis-lstm
Model Architecture: The sentiment analysis model employs an LSTM architecture.
Optimizer and Loss Function: Same as above.
Dataset: Same as above.
Accuracy: This model achieves an accuracy of 88%.

Installation:
To run this sentiment analysis tool, ensure you have TensorFlow installed. You can install it using pip:
pip install tensorflow