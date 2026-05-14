import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- CONFIGURATION ---
# Point this to your newly trained model from the 60/20/20 split
MODEL_PATH = 'computer_vision/models/digit_model_v3.h5'

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# 1. Fetch the Official Unseen MNIST Test Set (10,000 images)
print("Downloading official MNIST test dataset...")
(_, _), (x_test, y_true) = tf.keras.datasets.mnist.load_data()

# 2. Pre-process to match your model's expected input
# Reshape to (10000, 28, 28, 1) and normalize to 0-1
x_test_normalized = x_test.reshape(-1, 28, 28, 1) / 255.0

# 3. Predict on the Unseen Test Set
print("Running batch inference on 10,000 images...")
predictions = model.predict(x_test_normalized, verbose=0)
y_pred = np.argmax(predictions, axis=1)

# 4. Generate the Confusion Matrix
print("Generating plot...")
cm = confusion_matrix(y_true, y_pred)

# Set up the matplotlib figure size (10x8 is good for IEEE columns)
plt.figure(figsize=(10, 8))

# Create the heatmap using seaborn
# annot=True puts the numbers in the boxes
# fmt='d' formats them as standard integers
# cmap='Blues' gives it a clean, academic look
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=[str(i) for i in range(10)],
            yticklabels=[str(i) for i in range(10)])

# Add labels and titles
plt.xlabel('Predicted Digit', fontsize=12, fontweight='bold')
plt.ylabel('Actual Digit', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix: 10,000 Unseen Test Samples', fontsize=14, pad=15)

# 5. Save the figure for your report
# dpi=300 ensures it is print-quality for the IEEE format
plt.savefig('confusion_matrix_IEEE.png', dpi=300, bbox_inches='tight') 
print("Success! Saved 'confusion_matrix_IEEE.png' to your directory.")

# Display the plot on your screen
plt.show()