import depthai as dai
import cv2
import numpy as np

# --- CONFIGURATION ---
BLOB_PATH = "computer_vision/digit_model.blob" 
SENSITIVITY = 150       
MIN_AREA = 500          
MAX_AREA = 50000        

# 1. Initialize the Pipeline
pipeline = dai.Pipeline()

# 2. Define Nodes using 2.32.0.0 Namespace
cam = pipeline.create(dai.node.ColorCamera)
nn = pipeline.create(dai.node.NeuralNetwork)
xout_video = pipeline.create(dai.node.XLinkOut)
xin_nn = pipeline.create(dai.node.XLinkIn)
xout_nn = pipeline.create(dai.node.XLinkOut)

# 3. Configure Nodes
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setPreviewSize(640, 480) 

nn.setBlobPath(BLOB_PATH)

xout_video.setStreamName("video")
xin_nn.setStreamName("nn_in")
xout_nn.setStreamName("nn_out")

# 4. Link Nodes
cam.preview.link(xout_video.input)
xin_nn.out.link(nn.input)
nn.out.link(xout_nn.input)

# 5. Inference Loop
with dai.Device(pipeline) as device:
    print(f"--- PI 4 INFERENCE ONLINE | BUS SPEED: {device.getUsbSpeed()} ---")
    
    q_video = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    q_nn_in = device.getInputQueue(name="nn_in")
    q_nn_out = device.getOutputQueue(name="nn_out", maxSize=4, blocking=False)

    while True:
        frame = q_video.get().getCvFrame()
        
        # Preprocessing Pipeline (Computer Vision)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, SENSITIVITY, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_AREA < area < MAX_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                roi = thresh[y:y+h, x:x+w]
                if roi.size == 0: continue
                
                # Prepare ROI for the .blob model (28x28 grayscale)
                roi_res = cv2.resize(roi, (28, 28))
                nn_data = dai.ImgFrame()
                nn_data.setData(roi_res.flatten().astype(np.uint8))
                nn_data.setWidth(28)
                nn_data.setHeight(28)
                q_nn_in.send(nn_data)
                
                # Retrieve Hardware-Accelerated Prediction
                in_nn = q_nn_out.tryGet()
                if in_nn is not None:
                    prediction = in_nn.getFirstLayerFp16()
                    digit = np.argmax(prediction)
                    conf = np.max(prediction)
                    
                    if conf > 0.8:
                        # Draw Quantitative Results on Screen
                        label = f"ID: {digit} ({conf*100:.1f}%)"
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        print(f"Detected {digit} with {conf*100:.1f}% confidence")

        # Show Output Windows
        cv2.imshow("Main Feed (Quantitative)", frame)
        cv2.imshow("Robot Brain View (Qualitative)", cv2.resize(thresh, (400, 300)))

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
