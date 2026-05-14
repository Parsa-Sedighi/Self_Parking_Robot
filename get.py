import depthai as dai
import cv2
import numpy as np
import tensorflow as tf
import time

# --- CONFIGURATION ---
# Path to your original Keras model
MODEL_PATH = "computer_vision/digit_model2.h5" 

# 1. Load the TensorFlow model locally
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded local model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 2. Define Preprocessing Pipeline
def preprocess_for_ai(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize to 28x28 for the CNN input
    resized = cv2.resize(gray, (28, 28))
    
    # Normalize pixel values
    normalized = resized.astype("float32") / 255.0
    
    # Invert colors (alignment with training set)
    #normalized = 1.0 - normalized
    
    # Reshape for model input
    input_data = normalized.reshape(1, 28, 28, 1)
    return input_data, normalized

# 3. Setup OAK-D Camera Pipeline
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(640, 480) 
cam.setInterleaved(False)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam.preview.link(xout.input)

# 4. Main Inference Loop
with dai.Device(pipeline) as device:
    q_video = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    
    print("--- DESKTOP DIAGNOSTIC ONLINE ---")
    print("Press 'q' in the window to quit.")

    while True:
        in_frame = q_video.get()
        frame = in_frame.getCvFrame()
        
        # Run Preprocessing
        input_data, debug_view = preprocess_for_ai(frame)
        
        # Perform Inference on Desktop CPU
        prediction = model.predict(input_data, verbose=0)
        
        # Use simple numpy operations (no bracketed tags)
        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        
        # Visualize Results
        label = f"Digit: {digit} ({confidence*100:.1f}%)"
        color = (0, 255, 0) if confidence > 0.8 else (0, 0, 255)
        
        # Main Display
        cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("OAK-D Raw Feed", frame)
        
        # Qualitative Result: "What the Robot Sees"
        cv2.imshow("Robot Brain (28x28)", cv2.resize(debug_view, (280, 280)))
        
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()