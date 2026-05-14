import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# 1. Load your existing model
MODEL_PATH = 'computer_vision/models/digit_model_v3.h5'
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Fetch the Unseen MNIST Test Set
print("Downloading testing dataset...")
(_, _), (x_test, y_true) = tf.keras.datasets.mnist.load_data()

# 3. Define Blur Intensities (Kernel Sizes for Gaussian Blur)
# Odd numbers only: 1 (No blur), 3 (Slight), 5 (Medium), 7 (Heavy), 9 (Severe)
blur_kernels = [1, 3, 5, 7, 9]
accuracies = []

print("\nRunning Robustness Stress Test...")

# 4. Test the model at each level of distortion
for k in blur_kernels:
    blurred_images = []
    
    # Apply the specific blur level to all 10,000 images
    for img in x_test:
        if k == 1:
            blurred_img = img # Baseline
        else:
            blurred_img = cv2.GaussianBlur(img, (k, k), 0)
        blurred_images.append(blurred_img)
    
    # Pre-process for the model
    x_blurred_normalized = np.array(blurred_images).reshape(-1, 28, 28, 1) / 255.0
    
    # Predict and calculate accuracy
    predictions = model.predict(x_blurred_normalized, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    acc = np.mean(y_pred == y_true) * 100
    accuracies.append(acc)
    
    print(f"Kernel Size {k}x{k} -> Accuracy: {acc:.2f}%")

# 5. Generate the Line Graph
plt.figure(figsize=(8, 6))
plt.plot(blur_kernels, accuracies, marker='o', linestyle='-', color='indigo', linewidth=2.5, markersize=8)

# Formatting for academic standards
plt.title('System Robustness: Accuracy vs. Motion Blur', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Gaussian Blur Kernel Size (Higher = Blurry)', fontsize=12)
plt.ylabel('Test Set Accuracy (%)', fontsize=12)
plt.xticks(blur_kernels, labels=['1x1\n(Baseline)', '3x3', '5x5', '7x7', '9x9\n(Severe)'])
plt.ylim(0, 100)
plt.grid(True, linestyle='--', alpha=0.6)

# Save the figure
plt.savefig('blur_robustness_IEEE.png', dpi=300, bbox_inches='tight')
print("\nSuccess! Saved 'blur_robustness_IEEE.png' for your report.")
plt.show()