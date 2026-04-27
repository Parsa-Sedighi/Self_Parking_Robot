import depthai as dai
import cv2
import numpy as np
import tensorflow as tf

# 1. Load your trained model
try:
    model = tf.keras.models.load_model('computer_vision/digit_model2.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 2. Setup OAK-D Pipeline
pipeline = dai.Pipeline()

# Define Camera
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(640, 480)
cam.setInterleaved(False)
cam.setFps(30)

# Define Output
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam.preview.link(xout.input)

# 3. Start Device
with dai.Device(pipeline) as device:
    q = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    SCREEN_CENTER = 640 // 2
    
    print("Robot Vision System Online. Press 'q' to quit.")

    while True:
        # Get frame
        in_video = q.get()
        frame = in_video.getCvFrame()
        display_frame = frame.copy()

        # --- STEP A: IMAGE PRE-PROCESSING (The 'Messy Background' Fix) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Blur to reduce high-frequency noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # ADAPTIVE THRESHOLD: Looks at 11x11 pixel blocks to find local contrast
        # This ignores large shadows or bright spots on the wall
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # MORPHOLOGY: Close small gaps in the '2' and remove tiny speckles
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) # Remove dots
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) # Fill digit gaps

        # --- STEP B: CONTOUR FILTERING ---
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            
            # FILTER 1: Size (Is it big enough to be our parking number?)
            if 800 < area < 30000:
                
                # FILTER 2: Shape (Digits are usually taller than they are wide)
                if 0.2 < aspect_ratio < 1.0:
                    
                    # --- STEP C: PREDICTION ---
                    # Crop the candidate number from the binary 'thresh' image
                    digit_roi = thresh[y:y+h, x:x+w]
                    
                    # Resize to 28x28 and prepare for TensorFlow
                    digit_res = cv2.resize(digit_roi, (28, 28))
                    digit_res = digit_res / 255.0
                    digit_input = digit_res.reshape(1, 28, 28, 1)
                    
                    # Get AI Confidence
                    prediction = model.predict(digit_input, verbose=0)
                    digit = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    # Only act if the AI is reasonably sure
                    if confidence > 0.85:
                        # Draw green box
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Calculate Navigation Offset
                        obj_center_x = x + (w // 2)
                        offset = obj_center_x - SCREEN_CENTER
                        
                        # Navigation Logic
                        direction = "CENTER"
                        if offset < -60: direction = "<- TURN LEFT"
                        elif offset > 60: direction = "TURN RIGHT ->"
                        
                        # UI Overlay
                        label = f"ID:{digit} ({confidence*100:.0f}%) {direction}"
                        cv2.putText(display_frame, label, (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- STEP D: DISPLAY WINDOWS ---
        cv2.imshow("Robot Live View (Navigation)", display_frame)
        cv2.imshow("Robot Brain View (Threshold)", thresh)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()