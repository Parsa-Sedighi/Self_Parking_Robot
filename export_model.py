import tensorflow as tf

# Load your H5 model
model = tf.keras.models.load_model('computer_vision/digit_model2.h5')

# Save it as a SavedModel directory
# This creates a folder with 'assets', 'variables', and 'saved_model.pb'
model.save('computer_vision/saved_model_dir')

print("✅ SavedModel created at 'computer_vision/saved_model_dir'")