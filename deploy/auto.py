import depthai as dai
import cv2
import numpy as np
import serial
import time

# --- CONFIGURATION ---
BLOB_PATH = "models/digit_model_v3.blob" 
TARGET_DIGIT = 0       # The hardcoded mission objective
CONF_THRESHOLD = 80.0  # The 80% Agentic Fail-Safe Gate

MIN_AREA = 500    
MAX_AREA = 50000  

# --- SERIAL SETUP (RASPBERRY PI TO ARDUINO) ---
# Note: Check if your Arduino is on ttyACM0 or ttyUSB0
try:
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(2) # Allow Arduino to reboot upon connection
    print("✅ Serial connection to Arduino established.")
except Exception as e:
    print(f"🛑 SERIAL ERROR: {e}")
    print("Running in simulation mode (No motor commands will be sent).")
    arduino = None

def send_motor_command(command):
    """Sends a string command over UART to the Arduino"""
    if arduino:
        arduino.write((command + '\n').encode('utf-8'))
    print(f"🤖 AGENT COMMAND: {command}")

# --- 1. INITIALIZE PIPELINE ---
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

# --- 2. AUTONOMOUS EXECUTION LOOP ---
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
        
        # 1. OPTIMIZED PREPROCESSING
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
                
                # 2. RETRIEVE PREDICTION 
                in_nn = q_nn_out.get()
                prediction = in_nn.getFirstLayerFp16()
                digit = int(np.argmax(prediction))
                conf = float(np.max(prediction)) * 100
                
                # --- 3. AGENTIC STATE MACHINE ---
                if digit == TARGET_DIGIT and conf >= CONF_THRESHOLD:
                    target_found_this_frame = True
                    
                    # Calculate the center X coordinate of the bounding box
                    box_center_x = x + (w // 2)
                    screen_center_x = frame_width // 2
                    
                    # Determine positional error for steering
                    error = box_center_x - screen_center_x
                    deadzone = 80 # Pixel tolerance before turning
                    
                    if error < -deadzone:
                        send_motor_command("YAW_LEFT")
                    elif error > deadzone:
                        send_motor_command("YAW_RIGHT")
                    else:
                        # Target is centered and confirmed, approach!
                        send_motor_command("SURGE_FORWARD")
                    
                    # Break the contour loop so we don't send multiple commands in one frame
                    break 
                
                elif conf >= CONF_THRESHOLD:
                    # It confidently saw a number, but it's the WRONG number.
                    # Ignore it, do not trigger the search state immediately.
                    pass

        # If the loop finishes and no target was found, default to Search State
        if not target_found_this_frame:
            send_motor_command("SEARCH_SPIN")

        # Small delay to prevent flooding the Arduino serial buffer
        time.sleep(0.05)