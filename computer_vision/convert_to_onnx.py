import tensorflow as tf
import tf2onnx
import onnx

# 1. Load your trained model
model = tf.keras.models.load_model('computer_vision/digit_model2.h5')

# 2. Define the static input shape [Batch, Height, Width, Channels]
# This locks the model to [1, 28, 28, 1] to satisfy the OAK-D
spec = (tf.TensorSpec((1, 28, 28, 1), tf.float32, name="input"),)

# 3. Convert to ONNX
output_path = "computer_vision/model/digit_model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# 4. Save the file
onnx.save(model_proto, output_path)

print(f"✅ SUCCESS! Your model is now at: {output_path}")
