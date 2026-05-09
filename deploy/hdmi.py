import depthai as dai
import cv2
import numpy as np
from monkey_bot.motorAPI import Movement
import time

# --- CONFIG ---
BLOB_PATH = "computer_vision/digit_model.blob"
SENSITIVITY = 100       
MIN_AREA = 200          
MAX_AREA = 20000        
SCREEN_CENTER = 150     # 300px preview width / 2

# 1. Initialize Motors
try:
    robot = Movement()
    print("Motor API Initialized.")
except Exception as e:
    print(f"Motor API failed: {e}")
    robot = None

# 2. Setup Pipeline
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)

# Pi 3 Optimization: Low Res + High Compression
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setIspScale(1, 3)        # Shrink image 4x internally
cam.setPreviewSize(300, 300) # Tiny window for AI
cam.setInterleaved(False)
cam.setFps(10)               # Save USB 2.0 bandwidth

# Neural Network
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

# 3. Execution (Forced USB 2.0 Mode)
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
    q_video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    q_nn_in = device.getInputQueue(name="nn_in")
    q_nn_out = device.getOutputQueue(name="nn_out", maxSize=1, blocking=False)

    print(f"--- SSH SESSION ACTIVE | SPEED: {device.getUsbSpeed()} ---")
    print("Point the camera at a digit to begin...")

    try:
        while True:
            in_video = q_video.tryGet()
            if in_video is None: continue 

            frame = in_video.getCvFrame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, SENSITIVITY, 255, cv2.THRESH_BINARY)
            
            # NO cv2.imshow() here to prevent SSH crashes
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            found_this_frame = False
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if MIN_AREA < area < MAX_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    roi = thresh[y:y+h, x:x+w]
                    if roi.size == 0: continue
                    roi_res = cv2.resize(roi, (28, 28))
                    
                    # Send to OAK-D for AI Inference
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
                        
                        if conf > 0.8:
                            found_this_frame = True
                            offset = (x + (w // 2)) - SCREEN_CENTER
                            print(f"[{time.strftime('%H:%M:%S')}] MATCH: {digit} ({conf*100:.0f}%) | Offset: {offset}")
                            
                            # Auto-steering logic
                            if robot:
                                if offset < -40: 
                                    robot.yaw(mag=-0.3, time=0.05)
                                elif offset > 40: 
                                    robot.yaw(mag=0.3, time=0.05)
                                else: 
                                    robot.surge(mag=0.4, time=0.1)
            
            if not found_this_frame:
                time.sleep(0.01) # Keep CPU usage low

    except KeyboardInterrupt:
        print("\nShutting down safely...")
    finally:
        if robot:
            try:
                robot.stop(time=0)
                robot.close()
            except: pass
