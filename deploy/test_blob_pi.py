import depthai as dai
import cv2
import numpy as np
from monkey_bot.motorAPI import Movement
import time

# --- CONFIG ---
BLOB_PATH = "computer_vision/digit_model.blob"
DIAGNOSTIC_MODE = True 
# We set this to 127 - standard middle-of-the-road threshold
SENSITIVITY = 127       
MIN_AREA = 500          
MAX_AREA = 150000       # If it's bigger than half the screen, it's not a digit
SCREEN_CENTER = 320     

# 1. Initialize Motors
try:
    robot = Movement()
except:
    robot = None

# 2. Setup Pipeline
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setPreviewSize(640, 480)
cam.setInterleaved(False)
cam.setFps(30)

nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(BLOB_PATH)

xout_video = pipeline.create(dai.node.XLinkOut)
xout_video.setStreamName("video")
cam.preview.link(xout_video.input)

xin_nn = pipeline.create(dai.node.XLinkIn)
xin_nn.setStreamName("nn_in")
xin_nn.out.link(nn.input)

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn_out")
nn.out.link(xout_nn.input)

with dai.Device(pipeline) as device:
    q_video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    q_nn_in = device.getInputQueue(name="nn_in")
    q_nn_out = device.getOutputQueue(name="nn_out", maxSize=1, blocking=False)

    print("--- SYSTEM ONLINE | V11 RESET ---")

    try:
        while True:
            in_video = q_video.tryGet()
            if in_video is None: continue 

            frame = in_video.getCvFrame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # --- THE FIX: STANDARD BINARY ---
            # This turns BRIGHT things WHITE and DARK things BLACK.
            # If you cover the lens, the image is dark (0), so it stays black (0).
            _, thresh = cv2.threshold(gray, SENSITIVITY, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            target_found = False
            
            # Diagnostic: If the whole screen is white, tell us why
            white_pixel_count = np.sum(thresh == 255)
            if white_pixel_count > 250000:
                if DIAGNOSTIC_MODE:
                    print(f"DEBUG: Screen is too bright ({white_pixel_count} white pixels). Check exposure!")
                continue

            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                if MIN_AREA < area < MAX_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # AI Crop
                    roi = thresh[y:y+h, x:x+w]
                    if roi.size == 0: continue
                    roi_res = cv2.resize(roi, (28, 28))
                    
                    nn_data = dai.ImgFrame()
                    nn_data.setData(roi_res.flatten().astype(np.uint8))
                    nn_data.setWidth(28)
                    nn_data.setHeight(28)
                    q_nn_in.send(nn_data)
                    
                    in_nn = q_nn_out.tryGet()
                    if in_nn is not None:
                        prediction = in_nn.getFirstLayerFp16()
                        digit = np.argmax(prediction)
                        conf = np.max(prediction)

                        if conf > 0.75:
                            target_found = True
                            offset = (x + (w // 2)) - SCREEN_CENTER
                            print(f"*** MATCH: {digit} ({conf*100:.0f}%) Offset: {offset} ***")
                            
                            if not DIAGNOSTIC_MODE and robot:
                                if offset < -60: robot.yaw(mag=-0.3, time=0.05)
                                elif offset > 60: robot.yaw(mag=0.3, time=0.05)
                                else: robot.surge(mag=0.4, time=0.1)

            if not target_found and not DIAGNOSTIC_MODE and robot:
                robot.stop(time=0)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if robot:
            try:
                robot.stop(time=0)
                robot.close()
            except: pass
