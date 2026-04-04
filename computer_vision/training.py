""" Imports """
# Loads the heavy duty machine learning library to train the model
import tensorflow as tf
# Tools used to handle data tables (CSVs). Excel for Python
import pandas as pd
# Handles high performance math and multi-dimensional arrays (matrices)
import numpy as np
# Speceifc helper to change how numbers are labeled
from tensorflow.keras.utils import to_categorical
# Library to pop up a window and show image
import matplotlib.pyplot as plt
# Print the version
print(f"Tensorflow version: tf.__version__")

""" Data Pre-processing """
#Open and read file
train_df = pd.read_csv('computer_vision/train.csv')
# Pulls out the answer/label of each image between 0-9 
Y_train = train_df['label']
# Pulls out the Questions/pixels of each image, all 784 pixel values by dropping the label column
X_train = train_df.drop(labels=['label'], axis=1) 
# Prints and cofirms how many rows of data were sucessfully read.
print(f"Loaded {len(X_train)} training images.")



# 1. Normalize the data (Scale pixels from 0-255 to 0-1)
# This helps the model learn much faster
X_train = X_train / 255.0

# 2. Reshape to (Batch, Height, Width, Channels)
# The CSV gives a flat line of 784 pixels
# 28, 28 folds that line into a square
# 1 tells the model there is only 1 channel (grayscale)
# -1 is a wildcard that tells Python: Figure out how manu images there are based on total number of pixels.
X_train = X_train.values.reshape(-1, 28, 28, 1)

# 3. One-Hot Encode the labels
# Converts digit '3' into [0,0,0,1,0,0,0,0,0,0]
# The AI will decide which slot is the most likely, 10 because 0-9.
Y_train = to_categorical(Y_train, num_classes=10)
# Takes the first image and prepares it for display. The [:,:,0] just ensures we are looking at 2D plane of first color channel
plt.imshow(X_train[0][:,:,0], cmap='gray')
# To find the position of the 1. If the 1 is in the 3rd spot, it prints Label:3.
plt.title(f"Label: {np.argmax(Y_train[0])}")
# Command that pauses script and opens the winfoe on screen to show digit
plt.show()

""" Model Architecture """

from tensorflow.keras import layers, models

# 1. Build the Architecture
model = models.Sequential([
    # First "Feature Detector"
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second "Feature Detector"
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flattening into a 1D list for the final decision
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    
    # Output Layer: 10 neurons (one for each digit 0-9)
    layers.Dense(10, activation='softmax')
])

# 2. Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


print("END")

