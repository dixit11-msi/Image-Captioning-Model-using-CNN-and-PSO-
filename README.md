# Image-Captioning-Model-using-CNN-and-PSO-
INTRODUCTION:
Creating an image captioning model using Convolutional Neural Networks (CNN) and Particle Swarm Optimization (PSO) is an interesting and advanced approach. In this setup, CNNs are used for extracting features from images, and PSO can be employed to optimize various parameters or the model's architecture.
Overview:
CNN for Feature Extraction: CNNs are commonly used for feature extraction from images. The CNN processes the image to extract high-level features that are useful for describing the content of the image.
PSO for Optimization: Particle Swarm Optimization (PSO) can be used for hyperparameter tuning, optimizing the architecture of the neural network, or even fine-tuning the weights in certain scenarios. PSO is a global optimization algorithm inspired by social behavior, like bird flocking or fish schooling.
About the Dataset
The Flickr8k dataset is used for training and evaluating the image captioning system. It consists of 8,091 images, each with five captions describing the content of the image. The dataset provides a diverse set of images with multiple captions per image, making it suitable for training caption generation models.

Download the dataset from Kaggle and organize the files as follows:

flickr8k
Images
(image files)
captions.txt

CODE :
import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Embedding, Reshape, Input
from tensorflow.keras.models import Model
from pyswarm import pso  # Particle Swarm Optimization

# Load and preprocess image
def extract_features(r"C:\Users\dixit\Desktop\OIP (1).jpg")
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(229, 229, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu')
    ])
    
    image = load_img(image_path, target_size=(229, 229))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0
    features = model.predict(image)
    return features.flatten()

# Upload image and extract features
image_path = upload_image()
image_features = extract_features("C:\Users\dixit\Desktop\OIP (1).jpg")

# Example dataset
captions = ["A cat sitting on a mat", "A dog playing in the park", "A beautiful sunset over the ocean"]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(captions)
padded_sequences = pad_sequences(sequences, maxlen=10, padding='post')

def create_caption_model():
    inputs = Input(shape=(512,))  # CNN feature vector
    x = Dense(256, activation='relu')(inputs)
    
    x = Dropout(0.5)(x)
    x = Reshape((1, 256))(x)
    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(256)(x)
    outputs = Dense(vocab_size, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

model = create_caption_model()

# Define PSO for Hyperparameter Optimization
def objective_function(params):
    learning_rate, batch_size = params
    batch_size = int(batch_size)
    
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
    
    history = model.fit(padded_sequences, np.zeros((len(captions), vocab_size)), batch_size=batch_size, epochs=1, verbose=0)
    return -history.history['accuracy'][0]  # Minimize negative accuracy

lb = [0.0001, 8]  # Lower bounds for learning rate and batch size
ub = [0.01, 64]  # Upper bounds

best_params, _ = pso(objective_function, lb, ub)
print("Optimized Learning Rate and Batch Size:", best_params)


FUTURE SCOPE:
Fine-tuning: Experiment with fine-tuning the captioning model architecture and hyperparameters for improved performance.
Dataset Expansion: Incorporate additional datasets to increase the diversity and complexity of the trained model for example we can train the model on Flickr30k dataset.
Beam Search: Implement beam search decoding for generating multiple captions and selecting the best one.
User Interface Enhancements: Improve the Streamlit app's user interface and add features such as image previews and caption confidence scores.
Multilingual Captioning: Extend the model to generate captions in multiple languages by incorporating multilingual datasets.


