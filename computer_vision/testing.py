import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Load the trained model
model = tf.keras.models.load_model('computer_vision/digit_model.h5')

# 2. Load your new image
# Use cv2.IMREAD_GRAYSCALE to ensure it's 1 channel
img = cv2.imread('computer_vision/data/two.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not find 'test_digit.png'. Check your file path!")
else:
    # 3. Pre-process exactly like the training data
    # Resize just in case it wasn't 28x28
    img_resized = cv2.resize(img, (28, 28))
    
    # Normalize (0-1)
    img_normalized = img_resized / 255.0
    
    # Reshape to (1, 28, 28, 1) for the model
    input_data = img_normalized.reshape(1, 28, 28, 1)

    # 4. Predict
    prediction = model.predict(input_data)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    print(f"I am {confidence:.2f}% sure this is a {predicted_digit}")

    # 5. Show what the model "sees"
    plt.imshow(img_normalized, cmap='gray')
    plt.title(f"Predicted: {predicted_digit}")
    plt.show()