""" Imports """
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns # Added for Confusion Matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix # Added for Confusion Matrix

print(f"Tensorflow version: {tf.__version__}")

""" Data Pre-processing & 60/20/20 Splitting """
train_df = pd.read_csv('computer_vision/data/train.csv')

Y_all = train_df['label']
X_all = train_df.drop(labels=['label'], axis=1) 
print(f"Loaded {len(X_all)} total images.")

# 1. Normalize and Reshape ALL data
X_all = X_all / 255.0
X_all = X_all.values.reshape(-1, 28, 28, 1)
Y_all = to_categorical(Y_all, num_classes=10)

# 2. Manual 3-Way Split (60% Train, 20% Val, 20% Test)
X_temp, X_test, Y_temp, Y_test = train_test_split(X_all, Y_all, test_size=0.20, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=42)

print(f"Data Splitting Complete:")
print(f"Training: {len(X_train)} images (60%)")
print(f"Validation: {len(X_val)} images (20%)")
print(f"Testing: {len(X_test)} images (20%) - LOCKED IN VAULT")

""" Model Architecture """
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

""" Training with Augmentation """
train_datagen = ImageDataGenerator(
    rotation_range=15,    
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    zoom_range=0.2,       
    fill_mode='constant', 
    cval=0               
)

val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, Y_train, batch_size=32)
val_generator = val_datagen.flow(X_val, Y_val, batch_size=32)

print("\n--- Starting Training Phase ---")
history = model.fit(
    train_generator,      
    epochs=15,            
    validation_data=val_generator, 
    verbose=1             
)

import matplotlib.pyplot as plt

# The 'history' variable from model.fit() contains all the data
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Plotting Accuracy and Loss side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy Plot
ax1.plot(epochs, acc, 'b-', label='Training Accuracy', linewidth=2)
ax1.plot(epochs, val_acc, 'g--', label='Validation Accuracy', linewidth=2)
ax1.set_title('Model Accuracy Convergence', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.7)

# Loss Plot
ax2.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
ax2.plot(epochs, val_loss, 'r--', label='Validation Loss', linewidth=2)
ax2.set_title('Model Loss Convergence', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epochs', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.7)

plt.savefig('convergence_curves_IEEE.png', dpi=300, bbox_inches='tight')
print("Saved 'convergence_curves_IEEE.png'")
plt.show()

# Save Model
model.save('computer_vision/models/digit_model_v3.h5')
print("Model saved as digit_model_v3.h5")

""" Final Evaluation & Confusion Matrix """
print("\n--- Running Final Evaluation on Unseen 20% Test Set ---")

# Evaluate returns loss and accuracy
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test Set Accuracy: {test_acc*100:.2f}%\n")

# Run batch prediction for the confusion matrix
print("Generating Confusion Matrix Figure...")
predictions = model.predict(X_test, verbose=0)

# Convert one-hot encoded labels back to normal numbers (0-9)
y_pred_classes = np.argmax(predictions, axis=1)
y_true_classes = np.argmax(Y_test, axis=1)

# Generate Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Plotting
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=[str(i) for i in range(10)],
            yticklabels=[str(i) for i in range(10)])

plt.xlabel('Predicted Digit', fontsize=12, fontweight='bold')
plt.ylabel('Actual Digit', fontsize=12, fontweight='bold')
plt.title(f'Confusion Matrix on Kaggle 20% Hold-Out (Acc: {test_acc*100:.1f}%)', fontsize=14, pad=15)

# Save for your IEEE report
plt.savefig('confusion_matrix_IEEE.png', dpi=300, bbox_inches='tight') 
print("Success! Saved 'confusion_matrix_IEEE.png'.")

# Show on screen
plt.show()