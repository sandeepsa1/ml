import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Convert images to (64, 64) arrays
def preprocess_image(image_path, target_size=(64, 64)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0 # to keep values between 0 and 1
    return image_array

# Fetch images from folders and vectorize and store in a single variable
# Another variable stores labels. 1 for dogs. 0 for others
def generate_datavectors(data_len, path):
    X = []

    folder_path = path + "dogs"
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    image_files = image_files[:data_len//2]
    dogs_preprocessed = np.array([preprocess_image(os.path.join(folder_path, img_file)) for img_file in image_files])

    folder_path = path + "cats"
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    image_files = image_files[:data_len//2]
    cats_preprocessed = np.array([preprocess_image(os.path.join(folder_path, img_file)) for img_file in image_files])

    X = np.concatenate((dogs_preprocessed, cats_preprocessed), axis=0)
    Y = np.array([1] * len(dogs_preprocessed) + [0] * len(cats_preprocessed))

    return (X, Y)

def shuffle_data(X, Y):
    combined_data = list(zip(X, Y))
    np.random.shuffle(combined_data)
    combined_images_shuffled, labels_shuffled = zip(*combined_data)
    X = np.array(combined_images_shuffled)
    Y = np.array(labels_shuffled)

    return (X, Y)

def plotimageswithlabel(X, Y):
    fig, axes = plt.subplots(1, len(X), figsize=(20, 2))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(X[i])
        ax.set_title(Y[i])
        ax.axis('off')
    plt.show()