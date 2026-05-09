import depthai as dai
import cv2
import numpy as np
from monkey_bot.motorAPI import Movement
import time

# 1. Initialize Motors
robot = Movement()

# 2. Setup OAK-D Pipeline
pipeline = dai.Pipeline()

# --- FLEXIBLE NODE HELPER ---
# This looks for XLinkIn/Out in both possible library locations
def create_node(pipeline, node_type):
    try:
        return pipeline.create(getattr(dai.node, node_type))
    except AttributeError:
        return pipeline.create(getattr(dai, node_type))

# --- CAMERA SETUP ---
# Sticking with ColorCamera as it provides the 'preview' stream we need
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A) 
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setPreviewSize(640, 480)

# --- NEURAL NETWORK SETUP ---
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath("computer_vision/digit_model.blob") #

# --- COMMUNICATION LINKS (Using the helper) ---
xin_nn = create_node(pipeline, "XLinkIn") 
xin_nn.setStreamName("nn_in")
xin_nn.out.link(nn.input)

xout_nn = create_node(pipeline, "XLinkOut")
xout_nn.setStreamName("nn_out")
nn.out.link(xout_nn.input)

xout_video = create_node(pipeline, "XLinkOut")
xout_video.setStreamName("video")
cam.preview.link(xout_video.input)

# 3. Execution Loop
with dai.Device(pipeline) as device:
    q_video = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    q_nn_in = device.getInputQueue(name="nn_in")
    q_nn_out = device.getOutputQueue(name="nn_out", maxSize=4, blocking=False)
    
    SCREEN_CENTER = 640 // 2
    print("Self-Parking System Online. Scanning...")

    try:
        while True:
            frame = q_video.get().getCvFrame()
            
            # --- CV Pre-Processing ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            target_found = False
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                
                if 800 < area < 30000 and 0.2 < (float(w)/h) < 1.0:
                    # Resize and send to VPU
                    roi = thresh[y:y+h, x:x+w]
                    roi_res = cv2.resize(roi, (28, 28))
                    
                    nn_data = dai.ImgFrame()
                    nn_data.setFrame(roi_res)
                    nn_data.setWidth(28)
                    nn_data.setHeight(28)
                    q_nn_in.send(nn_data)
                    
                    # Receive prediction from OAK-D
                    res = q_nn_out.get()
                    prediction = res.getFirstLayerFp16()
                    digit = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    if confidence > 0.85:
                        target_found = True
                        offset = (x + w // 2) - SCREEN_CENTER
                        
                        if offset < -60:
                            print(f"ID:{digit} Left - Adjusting...")
                            robot.yaw(mag=-0.3, time=0.1)
                        elif offset > 60:
                            print(f"ID:{digit} Right - Adjusting...")
                            robot.yaw(mag=0.3, time=0.1)
                        else:
                            print(f"ID:{digit} Centered - Parking")
                            robot.surge(mag=0.4, time=0.2)
            
            if not target_found:
                robot.stop(time=0)

            # Headless: We keep the frame processing but remove the window display

    except KeyboardInterrupt:
        print("\nStopping...")
        robot.stop(time=0)
    finally:
        robot.close()
