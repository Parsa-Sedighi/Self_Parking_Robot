import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# 1. Load the dataset
# Adjust the path if your CSV is in a different folder
train_df = pd.read_csv('computer_vision/data/train.csv')

# 2. Extract labels and pixels
labels = train_df['label'].values
pixels = train_df.drop(labels=['label'], axis=1).values

# 3. Setup the visualization grid
plt.figure(figsize=(12, 10))
plt.suptitle("Random Samples from Training Data", fontsize=16)

# 4. Pick 20 random indices and plot them
for i in range(20):
    idx = random.randint(0, len(pixels) - 1)
    
    # Reshape the flat 784 pixels back into 28x28
    img = pixels[idx].reshape(28, 28)
    
    plt.subplot(4, 5, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {labels[idx]}")
    plt.axis('off') # Hide the X/Y coordinate numbers

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()