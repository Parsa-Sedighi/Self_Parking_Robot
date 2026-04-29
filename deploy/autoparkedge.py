import depthai as dai
import cv2
import numpy as np
from monkey_bot.motorAPI import Movement

# 1. Setup Motors
robot = Movement()

# 2. Setup OAK-D Pipeline
pipeline = dai.Pipeline()

# Define Camera
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(640, 480)
cam.setInterleaved(False)

# Define Neural Network Node (The Brain)
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath("computer_vision/digit_model.blob") # Your .blob file

# Define Input/Output Links
xin_nn = pipeline.create(dai.node.XLinkIn) # Mailbox to send crops to OAK-D
xin_nn.setStreamName("nn_in")
xin_nn.out.link(nn.input)

xout_nn = pipeline.create(dai.node.XLinkOut) # Queue to get AI results back
xout_nn.setStreamName("nn_out")
nn.out.link(xout_nn.input)

xout_video = pipeline.create(dai.node.XLinkOut)
xout_video.setStreamName("video")
cam.preview.link(xout_video.input)

# 3. Start Device
with dai.Device(pipeline) as device:
    q_video = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    q_nn_in = device.getInputQueue(name="nn_in") # Used to send crops
    q_nn_out = device.getOutputQueue(name="nn_out", maxSize=4, blocking=False) # Get answers
    
    SCREEN_CENTER = 640 // 2
    print("Edge-AI Navigation Online. Offloading math to OAK-D...")

    while True:
        frame = q_video.get().getCvFrame()
        
        # --- Pre-Processing (CV Logic) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            if 800 < area < 30000 and 0.2 < (float(w)/h) < 1.0:
                # --- EDGE AI PREDICTION ---
                # Crop and resize the candidate digit
                roi = thresh[y:y+h, x:x+w]
                roi_res = cv2.resize(roi, (28, 28))
                
                # Send crop to OAK-D for inference
                nn_data = dai.ImgFrame()
                nn_data.setFrame(roi_res)
                nn_data.setWidth(28)
                nn_data.setHeight(28)
                q_nn_in.send(nn_data)
                
                # Get Result from OAK-D
                res = q_nn_out.get()
                prediction = res.getFirstLayerFp16()
                digit = np.argmax(prediction)
                confidence = np.max(prediction)
                
                if confidence > 0.85:
                    obj_center_x = x + (w // 2)
                    offset = obj_center_x - SCREEN_CENTER
                    
                    # Navigation Logic
                    if offset < -60: robot.yaw(mag=-0.3, time=0.1)
                    elif offset > 60: robot.yaw(mag=0.3, time=0.1)
                    else: robot.surge(mag=0.4, time=0.2)

        cv2.imshow("Navigation", frame)
        if cv2.waitKey(1) == ord('q'):
            break