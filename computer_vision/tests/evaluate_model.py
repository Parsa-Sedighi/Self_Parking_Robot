import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report




# --- CONFIGURATION ---
# Point this to your Kaggle train.csv file!
DATA_PATH = 'computer_vision/data/train.csv' 
MODEL_PATH = 'computer_vision/models/digit_model2.h5'

# 1. Load the trained model
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Load the labeled dataset
print(f"Loading labeled dataset from {DATA_PATH}...")
try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Could not find '{DATA_PATH}'. Please check your path.")
    exit()

# 3. Data Splitting (IEEE Paper Standard)
# Isolate the labels (Column 0) and the pixels (Columns 1-784)
y_all = data.iloc[:, 0].values
x_all = data.iloc[:, 1:].values

# Split 20% of the data specifically for testing
_, x_test, _, y_true = train_test_split(x_all, y_all, test_size=0.2, random_state=42)
print(f"Reserved {len(x_test)} images for the test evaluation.")

# 4. Pre-process exactly like the training data
# Reshape from flat 784 pixels to (N, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Normalize (0-1)
x_test_normalized = x_test / 255.0

# 5. Predict on the Test Set
print("Running batch inference...")
predictions = model.predict(x_test_normalized, verbose=0)
y_pred = np.argmax(predictions, axis=1)
confidences = np.max(predictions, axis=1)

# 6. Calculate Core Metrics
accuracy = np.mean(y_pred == y_true) * 100
avg_confidence = np.mean(confidences) * 100

print("\n--- QUANTITATIVE RESULTS ---")
print(f"Overall Accuracy:   {accuracy:.2f}%")
print(f"Average Confidence: {avg_confidence:.2f}%")
print("\nClassification Report (Precision, Recall, F1-Score):")
print(classification_report(y_true, y_pred))

# 7. Generate Confusion Matrix Plot 
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Value', fontsize=12)
plt.ylabel('Actual Value', fontsize=12)
plt.title('Confusion Matrix on Test Set (20% Holdout)', fontsize=14)

plt.savefig('confusion_matrix_figure.png', dpi=300, bbox_inches='tight') 
print("\nSaved 'confusion_matrix_figure.png' for your report.")
plt.show()

# 8. Extract Data for Failure Analysis Section
incorrect_indices = np.where(y_pred != y_true)[0]
print(f"Total Misclassified Images: {len(incorrect_indices)}")

if len(incorrect_indices) > 0:
    print("Plotting sample failure cases...")
    fig, axes = plt.subplots(1, min(5, len(incorrect_indices)), figsize=(15, 3))
    if len(incorrect_indices) == 1: axes = [axes]
    
    for i, idx in enumerate(incorrect_indices[:5]):
        img = x_test_normalized[idx].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"True: {y_true[idx]}\nPred: {y_pred[idx]}\nConf: {confidences[idx]*100:.1f}%")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig('failure_cases_figure.png', dpi=300, bbox_inches='tight')
    print("Saved 'failure_cases_figure.png' for your report.")
    plt.show()