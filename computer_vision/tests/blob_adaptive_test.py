import depthai as dai
import cv2
import numpy as np

# --- CONFIGURATION ---
BLOB_PATH = "computer_vision/models/digit_model_v3.blob" 
MIN_AREA = 500    # Ignore tiny noise (pebbles, glare)
MAX_AREA = 50000  # Ignore massive objects (walls, parking lines)

# 1. Initialize Pipeline
pipeline = dai.Pipeline()

# 2. Define Nodes 
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setPreviewSize(640, 480)
cam.setInterleaved(False)

# The Neural Network Node
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(BLOB_PATH)

# I/O Links
xin_nn = pipeline.create(dai.node.XLinkIn)
xin_nn.setStreamName("nn_in")
xin_nn.out.link(nn.input)

xout_video = pipeline.create(dai.node.XLinkOut)
xout_video.setStreamName("video")
cam.preview.link(xout_video.input)

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn_out")
nn.out.link(xout_nn.input)

# 3. Diagnostic Execution Loop
with dai.Device(pipeline) as device:
    print("\n--- DYNAMIC .BLOB DIAGNOSTIC ONLINE ---")
    print("Hold a number in the frame. Press 'q' to quit.")
    
    q_video = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    q_nn_in = device.getInputQueue(name="nn_in")
    q_nn_out = device.getOutputQueue(name="nn_out", maxSize=4, blocking=False)

    while True:
        in_frame = q_video.get()
        frame = in_frame.getCvFrame()
        
        # 1. ADAPTIVE PREPROCESSING
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Otsu's method automatically calculates the optimal threshold value
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # --- NEW: MORPHOLOGICAL EROSION ---
        # Thins out the white lines to prevent the top of a '6' from bleeding into an '8'
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        
        # 2. DYNAMIC CONTOUR DETECTION
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a blank image for the debug window just in case nothing is detected
        display_roi = np.zeros((28, 28), dtype=np.uint8)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Filter out noise and giant blobs
            if MIN_AREA < area < MAX_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # --- ASPECT RATIO FILTERING ---
                # Reject anything that is too wide (like a square) or too thin (like a line)
                aspect_ratio = w / float(h)
                if aspect_ratio < 0.2 or aspect_ratio > 0.85:
                    continue 
                
                # --- NEW: INCREASED PADDING ---
                # Increased pad to 25 to capture more black space. 
                # This prevents the number from getting "squished" during the 28x28 resize.
                pad = 25
                y1, y2 = max(0, y - pad), min(frame.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(frame.shape[1], x + w + pad)
                
                # Extract the dynamic Region of Interest (ROI)
                roi = thresh[y1:y2, x1:x2]
                if roi.size == 0: continue
                
                # Resize for the Neural Network
                roi_res = cv2.resize(roi, (28, 28))
                display_roi = roi_res # Save to show in our debug window
                
                # Package and send to the VPU
                nn_data = dai.ImgFrame()
                nn_data.setData(roi_res.flatten().astype(np.uint8))
                nn_data.setWidth(28)
                nn_data.setHeight(28)
                q_nn_in.send(nn_data)
                
                # Retrieve Edge-AI Prediction 
                in_nn = q_nn_out.get()
                prediction = in_nn.getFirstLayerFp16()
                digit = int(np.argmax(prediction))
                conf = float(np.max(prediction))
                
                # Capped confidence display to prevent unrealistic 100.0% readouts
                display_conf = min(conf * 100, 99.9)
                
                # 3. VISUAL FEEDBACK 
                color = (0, 255, 0) if display_conf > 80.0 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Put the text right above the bounding box
                label = f"Target: {digit} ({display_conf:.1f}%)"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Render Windows
        cv2.imshow("Desktop .blob Hardware Test", frame)
        cv2.imshow("Otsu Threshold & Erosion View", cv2.resize(thresh, (320, 240))) 
        cv2.imshow("VPU Input (28x28)", cv2.resize(display_roi, (200, 200)))

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()