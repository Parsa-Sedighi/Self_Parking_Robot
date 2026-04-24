import depthai as dai
import cv2
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('computer_vision/digit_model2.h5')

pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(640, 480)
cam.setInterleaved(False)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam.preview.link(xout.input)

with dai.Device(pipeline) as device:
    q = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    
    while True:
        frame = q.get().getCvFrame()
        
        # 1. Prepare the "Region of Interest" (ROI) - a 200x200 square in the center
        roi = frame[140:340, 220:420]
        
        # 2. Pre-process for the model (Grayscale -> Resize -> Normalize)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        normalized = resized / 255.0
        input_data = np.reshape(normalized, (1, 28, 28, 1))
        
        # 3. Predict
        prediction = model.predict(input_data, verbose=0)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # 4. Draw UI
        cv2.rectangle(frame, (220, 140), (420, 340), (0, 255, 0), 2)
        cv2.putText(frame, f"Digit: {digit} ({confidence*100:.1f}%)", (220, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Robot AI Deployment", frame)
        if cv2.waitKey(1) == ord('q'): break