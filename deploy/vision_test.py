import depthai as dai
import cv2
import numpy as np
import time

# 1. Setup OAK-D Pipeline (Version 2.24.0 Standard)
pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setBoardSocket(dai.CameraBoardSocket.RGB) 
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setPreviewSize(640, 480)

nn = pipeline.createNeuralNetwork()
nn.setBlobPath("computer_vision/digit_model.blob")

xin_nn = pipeline.createXLinkIn()
xin_nn.setStreamName("nn_in")
xin_nn.out.link(nn.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn_out")
nn.out.link(xout_nn.input)

xout_video = pipeline.createXLinkOut()
xout_video.setStreamName("video")
cam.preview.link(xout_video.input)

# 2. Diagnosis Loop
with dai.Device(pipeline) as device:
    q_video = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    q_nn_in = device.getInputQueue(name="nn_in")
    q_nn_out = device.getOutputQueue(name="nn_out", maxSize=4, blocking=False)
    
    print("--- VISION DIAGNOSIS MODE ---")
    print("Scanning for any shapes... (Press Ctrl+C to stop)")

    try:
        while True:
            in_video = q_video.get()
            frame = in_video.getCvFrame()
            
            # CV Pre-Processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            found_anything = False
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h
                
                # Check for ANY reasonably sized object
                if 100 < area < 100000:
                    found_anything = True
                    
                    # Send to VPU for identification
                    roi = thresh[y:y+h, x:x+w]
                    roi_res = cv2.resize(roi, (28, 28))
                    
                    nn_data = dai.ImgFrame()
                    nn_data.setFrame(roi_res)
                    nn_data.setWidth(28)
                    nn_data.setHeight(28)
                    q_nn_in.send(nn_data)
                    
                    res = q_nn_out.get()
                    prediction = res.getFirstLayerFp16()
                    digit = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    # REPORT EVERYTHING to the terminal
                    print(f"[BLOB FOUND] Area: {int(area)} | Ratio: {aspect_ratio:.2f} | AI thinks it's a '{digit}' ({confidence*100:.1f}%)")

            if not found_anything:
                print("...Searching (No shapes detected)...")
            
            time.sleep(0.5) # Slow down the text so you can read it

    except KeyboardInterrupt:
        print("\nDiagnosis Complete.")
