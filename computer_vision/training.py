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

from tensorflow.keras import layers, models

# Print the version
print(f"Tensorflow version: tf.__version__")

""" Data Pre-processing """

#Open and read file
train_df = pd.read_csv('computer_vision/data/train.csv')
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

# 1. Creates a linear stack of layers. Information enters at the top and flows through each layer
# in order until it reaches the output.
model = models.Sequential([
    # First "Feature Detector": 32 different filters/feature detector
    # Each filter is a 3x3 pixel window that slides across the image looking for patterns like edges or corners
    # relu = The "Rectified Linear Unit" turns negative numbers to zero, helping the model learn non-linear patters (like the difference between straight line and a curve)
    # The model will expect the first layer to be 28 x 28 grayscale image
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Downsampler, looks at 2 x 2 pixel blocks and keeps only the highest value. It shrinks the image size by
    # half, making the model faster and focusing on the most important features.
    layers.MaxPooling2D((2, 2)),
    
    # Second "Feature Detector": Since first layer found basic edges, this layer can now
    # combine those edges to find more complex shapes (like the loops in a '6' or an '8').
    layers.Conv2D(64, (3, 3), activation='relu'),
    # Shrinks the data again, the image is very small but packed with high level information about shape it found
    layers.MaxPooling2D((2, 2)),
    
    # Flattening 2D into a 1D list for the final decision. You have to "flatten" the data
    # before you can feed it into the final decision making neurons
    layers.Flatten(),
    # A fully connected layer: these 64 neurons look at all the features found by the previous layers
    # and start voting on which digit it might be.
    layers.Dense(64, activation='relu'),
    
    # Output Layer: 10 neurons (one for each digit 0-9)
    #softmax turns the output into probabilities. Instead of just saying "it's a 3," 
    # it might say: "95%" chance it's a 3, 2% chance it's an 8, 3% chance it's a 2.
    layers.Dense(10, activation='softmax')
])

# 2. Compile
# Tells the model how to improve itself during training
model.compile(optimizer='adam', # the "driver". It's an algorithm that decides how much to change the weight of the neurons based on errors.
              loss='categorical_crossentropy', # The judge. This formula calculates exactly how "wrong" the model was. since to_categorial was used earlier, this is the standar math for multi-class problems
              metrics=['accuracy']) # Tells the TensorFlow to print out the percentage of correct guessses during trainning so you can track progress
#Prints a table showing every layer, the total number of "Parameters" (the indivdual weights the model has to learn)
model.summary()

""" Training """

# 3. Train the Model
# Training command = model.fit

history = model.fit(
    X_train, Y_train, 
    # Study rounds, the model will look at all 42,000 images 10 seperatr times. Usually, accuracy goes up with each epoch.
    epochs=10,           # How many times to go through the whole dataset
    # Instead of looking at all images at once, the model looks at 32, calculates the error, adjusts its brain and move to the next 32, this is faster and uses less memory
    batch_size=32,       # How many images to look at before updating weights
    # The model hides 10% of the data from itself. After every epoch, it tests itself on this "hidden" data to see if it's actually learning patterns or just memorizing the pictures, AKA overfitting
    validation_split=0.1 # Save 10% of data to test itself during training
)
 # 4. Saving the model
#model.save('computer_vision/digit_model.h5')
#print("Model saves as digital_model.h5")

model = tf.keras.models.load_model('computer_vision/digit_model.h5')
model.save('computer_vision/digit_model_saved') # This creates a folder
