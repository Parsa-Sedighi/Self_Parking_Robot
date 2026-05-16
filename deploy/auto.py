import depthai as dai
import cv2
import numpy as np
import time
import sys

# --- IMPORT YOUR CUSTOM HARDWARE API ---
from monkey_bot.motorAPI import Movement

# --- CONFIGURATION ---
# 1. THE ABSOLUTE PATH FIX
BLOB_PATH = "/home/parsapi/Self_Parking_Robot/computer_vision/models/digit_model_v3.blob" 

# 2. THE HARDCODED TARGET (Change this number to hunt a different digit)
TARGET_DIGIT = 0       

CONF_THRESHOLD = 80.0  
MIN_AREA = 500    
MAX_AREA = 50000  

# --- 1. INITIALIZE MOTORS ---
try:
    robot = Movement()
except Exception as e:
    print(f"🛑 SERIAL ERROR: {e}")
    print("Cannot find Arduino. Exiting.")
    sys.exit(1)

# --- 2. INITIALIZE PIPELINE ---
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setPreviewSize(640, 480)
cam.setInterleaved(False)

nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(BLOB_PATH)

xin_nn = pipeline.create(dai.node.XLinkIn)
xin_nn.setStreamName("nn_in")
xin_nn.out.link(nn.input)

xout_video = pipeline.create(dai.node.XLinkOut)
xout_video.setStreamName("video")
cam.preview.link(xout_video.input)

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn_out")
nn.out.link(xout_nn.input)

# --- 3. AUTONOMOUS EXECUTION LOOP ---
try:
    with dai.Device(pipeline) as device:
        print("\n" + "="*50)
        print(f"🚀 EDGE-AI NAVIGATION AGENT ONLINE")
        print(f"🎯 MISSION: Locate and Approach Target [{TARGET_DIGIT}]")
        print("="*50 + "\n")
        
        q_video = device.getOutputQueue(name="video", maxSize=4, blocking=False)
        q_nn_in = device.getInputQueue(name="nn_in")
        q_nn_out = device.getOutputQueue(name="nn_out", maxSize=4, blocking=False)

        while True:
            in_frame = q_video.get()
            frame = in_frame.getCvFrame()
            frame_width = frame.shape[1]
            
            # OPTIMIZED PREPROCESSING
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.erode(thresh, kernel, iterations=1)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            target_found_this_frame = False

            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                if MIN_AREA < area < MAX_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Aspect Ratio Filter
                    aspect_ratio = w / float(h)
                    if aspect_ratio < 0.1 or aspect_ratio > 0.9:
                        continue 
                    
                    roi = thresh[y:y+h, x:x+w]
                    if roi.size == 0: continue
                    
                    # Square Canvas Centering
                    max_edge = max(w, h)
                    pad = int(max_edge * 0.25) 
                    canvas_size = max_edge + (pad * 2)
                    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
                    
                    x_offset = (canvas_size - w) // 2
                    y_offset = (canvas_size - h) // 2
                    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = roi
                    
                    roi_res = cv2.resize(canvas, (28, 28))
                    
                    # Hardware Alignment
                    nn_data = dai.ImgFrame()
                    nn_data.setData(roi_res.flatten().astype(np.uint8))
                    nn_data.setWidth(28)
                    nn_data.setHeight(28)
                    q_nn_in.send(nn_data)
                    
                    # RETRIEVE PREDICTION 
                    in_nn = q_nn_out.get()
                    prediction = in_nn.getFirstLayerFp16()
                    digit = int(np.argmax(prediction))
                    conf = float(np.max(prediction)) * 100
                    
                    # --- AGENTIC STATE MACHINE ---
                    if digit == TARGET_DIGIT and conf >= CONF_THRESHOLD:
                        target_found_this_frame = True
                        
                        box_center_x = x + (w // 2)
                        screen_center_x = frame_width // 2
                        
                        error = box_center_x - screen_center_x
                        deadzone = 80 
                        
                        # Note: delay=0 ensures the camera loop is not blocked during tracking!
                        if error < -deadzone:
                            robot.yaw(-0.4, 0) # Spin Left
                        elif error > deadzone:
                            robot.yaw(0.4, 0)  # Spin Right
                        else:
                            robot.surge(0.6, 0) # Target Centered, Move Forward
                        
                        break 

            # --- NEW: PULSE SCANNING ---
            if not target_found_this_frame:
                # 1. Turn slightly for a fraction of a second
                robot.yaw(0.4, 0.1) 
                
                # 2. Stop completely to eliminate motion blur
                robot.stop(0.2)
                
                # The script will now capture the next frame while completely still!

            # Keep the loop from completely maxing out the CPU
            time.sleep(0.01)

except KeyboardInterrupt:
    print("\n🛑 Manual override triggered. Shutting down system...")

finally:
    # --- FAIL-SAFE SHUTDOWN ---
    try:
        robot.stop(0)
        robot.close()
    except:
        pass
