import depthai as dai
import cv2
import numpy as np
import tensorflow as tf

# 1. Load the "Brain"
model = tf.keras.models.load_model('computer_vision/digit_model2.h5')

# 2. Setup OAK-D Pipeline
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(640, 480)
cam.setInterleaved(False)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam.preview.link(xout.input)

with dai.Device(pipeline) as device:
    q = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    SCREEN_CENTER = 640 // 2

    while True:
        frame = q.get().getCvFrame()
        debug_frame = frame.copy()
        
        # --- IMAGE PROCESSING TO FIND THE NUMBER ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Threshold turns the image black and white based on contrast
        _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV) 
        
        # Find "Contours" (shapes)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter by size: ignore tiny dust and huge background objects
            if 500 < area < 20000:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Extract the number from the frame
                digit_roi = thresh[y:y+h, x:x+w]
                
                # Make it a square for the model
                pad = 10
                digit_res = cv2.resize(digit_roi, (28, 28))
                digit_res = digit_res / 255.0
                digit_input = digit_res.reshape(1, 28, 28, 1)
                
                # Predict
                prediction = model.predict(digit_input, verbose=0)
                digit = np.argmax(prediction)
                confidence = np.max(prediction)
                
                if confidence > 0.8: # Only act if we are sure
                    # Draw the Bounding Box
                    cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Calculate Navigation Logic
                    obj_center_x = x + (w // 2)
                    offset = obj_center_x - SCREEN_CENTER
                    
                    direction = "CENTER"
                    if offset < -50: direction = "TURN LEFT"
                    elif offset > 50: direction = "TURN RIGHT"
                    
                    cv2.putText(debug_frame, f"ID: {digit} | {direction}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Robot Navigation View", debug_frame)
        cv2.imshow("What the Robot Sees (Threshold)", thresh)
        
        if cv2.waitKey(1) == ord('q'): break

cv2.destroyAllWindows()