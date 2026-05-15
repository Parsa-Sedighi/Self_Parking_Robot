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

# I/O Links (Using Host-Side Preprocessing)
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
        
        # Otsu's method calculates the optimal lighting threshold dynamically
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological Erosion: Thins out white lines so '6's don't bleed into '8's
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        
        # 2. DYNAMIC CONTOUR DETECTION
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Blank image for the debug window just in case nothing is detected
        display_roi = np.zeros((28, 28), dtype=np.uint8)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Filter out noise and giant blobs
            if MIN_AREA < area < MAX_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # --- ASPECT RATIO FILTERING ---
                # Lower limit is 0.1 to ensure we don't ignore thin '1's
                aspect_ratio = w / float(h)
                if aspect_ratio < 0.1 or aspect_ratio > 0.9:
                    continue 
                
                # Extract the exact tight bounding box around the number
                roi = thresh[y:y+h, x:x+w]
                if roi.size == 0: continue
                
                # --- SQUARE CANVAS CENTERING LOGIC ---
                # Find the longest edge to create a perfect square
                max_edge = max(w, h)
                
                # Add 25% padding relative to the number's size so it doesn't touch the borders
                pad = int(max_edge * 0.25) 
                canvas_size = max_edge + (pad * 2)
                
                # Create a pure black canvas
                canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
                
                # Calculate the exact offset to paste the number in the dead center
                x_offset = (canvas_size - w) // 2
                y_offset = (canvas_size - h) // 2
                
                # Paste the un-stretched number into the center of the black square
                canvas[y_offset:y_offset+h, x_offset:x_offset+w] = roi
                
                # Finally, resize the perfectly square canvas down to 28x28 for the VPU
                roi_res = cv2.resize(canvas, (28, 28))
                display_roi = roi_res # Save to show in our debug window
                
                # --- HARDWARE ALIGNMENT ---
                # Package and send to the VPU as an ImgFrame (enforces uint8 format)
                nn_data = dai.ImgFrame()
                nn_data.setData(roi_res.flatten().astype(np.uint8))
                nn_data.setWidth(28)
                nn_data.setHeight(28)
                q_nn_in.send(nn_data)
                
                # 3. RETRIEVE PREDICTION 
                in_nn = q_nn_out.get()
                prediction = in_nn.getFirstLayerFp16()
                digit = int(np.argmax(prediction))
                conf = float(np.max(prediction))
                
                # Display raw, true confidence (no 99.9% artificial cap)
                display_conf = conf * 100
                
                # 4. VISUAL FEEDBACK 
                color = (0, 255, 0) if display_conf > 80.0 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Put the text right above the bounding box
                label = f"Target: {digit} ({display_conf:.1f}%)"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Render Windows
        cv2.imshow("Desktop .blob Hardware Test", frame)
        cv2.imshow("Otsu Threshold & Erosion View", cv2.resize(thresh, (320, 240))) 
        cv2.imshow("VPU Input (28x28)", cv2.resize(display_roi, (200, 200)))

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()