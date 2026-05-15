import depthai as dai
import cv2
import numpy as np

# --- CONFIGURATION ---
BLOB_PATH = "computer_vision/models/digit_model_v3.blob" 

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
    print(f"--- DESKTOP .BLOB DIAGNOSTIC ONLINE ---")
    print("Hold a number inside the green box. Press 'q' to quit.")
    
    q_video = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    q_nn_in = device.getInputQueue(name="nn_in")
    q_nn_out = device.getOutputQueue(name="nn_out", maxSize=4, blocking=False)

    while True:
        in_frame = q_video.get()
        frame = in_frame.getCvFrame()
        h, w = frame.shape[:2]
        
        # Define a static targeting box in the center of the screen
        box_size = 150
        y1, y2 = h//2 - box_size, h//2 + box_size
        x1, x2 = w//2 - box_size, w//2 + box_size
        
        # Extract the center Region of Interest (ROI)
        roi = frame[y1:y2, x1:x2]
        
        # Preprocessing: Grayscale -> Threshold -> Resize to 28x28
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        roi_res = cv2.resize(thresh, (28, 28))
        
        # Package and send to the VPU
        nn_data = dai.ImgFrame()
        nn_data.setData(roi_res.flatten().astype(np.uint8))
        nn_data.setWidth(28)
        nn_data.setHeight(28)
        q_nn_in.send(nn_data)
        
        # Retrieve Edge-AI Prediction
        in_nn = q_nn_out.tryGet()
        if in_nn is not None:
            prediction = in_nn.getFirstLayerFp16()
            digit = int(np.argmax(prediction))
            conf = float(np.max(prediction))
            
            # Visual Feedback
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"VPU Output: {digit} ({conf*100:.1f}%)"
            color = (0, 255, 0) if conf > 0.8 else (0, 0, 255)
            cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Render Windows
        cv2.imshow("Desktop .blob Hardware Test", frame)
        cv2.imshow("VPU Input (28x28)", cv2.resize(roi_res, (200, 200))) # Scaled up for your visibility

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()