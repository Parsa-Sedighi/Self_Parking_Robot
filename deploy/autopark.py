import depthai as dai
import numpy as np
from monkey_bot.motorAPI import Movement
import time

robot = Movement()

pipeline = dai.Pipeline()

# 1. Use the new 'Camera' node instead of 'ColorCamera'
cam = pipeline.create(dai.node.Camera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A) # Usually CAM_A is the RGB camera
cam.setSize(640, 480)

# 2. Setup the Neural Network
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath("computer_vision/digit_model.blob")

# 3. Create XLinkOut (Ensure it's using the correct attribute)
xout_video = pipeline.create(dai.node.XLinkOut)
xout_video.setStreamName("video")

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")

# 4. Linking
cam.video.link(xout_video.input)
# Note: For the NN input, we usually use the 'preview' or 'video' output
cam.video.link(nn.input) 
nn.out.link(xout_nn.input)
# 3. Execution Loop
with dai.Device(pipeline) as device:
    # maxSize=1 prevents command backlog for low-latency parking
    q = device.getOutputQueue(name="detections", maxSize=1, blocking=False)
    
    print("Robot Online. Searching for parking digit...")

    try:
        while True:
            in_nn = q.get()
            data = in_nn.getFirstLayerFp16()
            digit = np.argmax(data)
            confidence = data[digit]

            # Threshold for action (adjust based on lighting)
            if confidence > 0.85:
                print(f"Target Found: {digit} ({confidence:.2f})")
                
                if digit == 1: # Let's say '1' means 'Park Here'
                    print("Parking spot confirmed. Surging...")
                    robot.surge(mag=0.5, time=2) # Move forward half-speed for 2s
                    robot.stop(time=1)
                    break 
                
                elif digit == 2: # Let's say '2' means 'Rotate to adjust'
                    print("Adjusting alignment...")
                    robot.yaw(mag=0.3, time=0.5) # Slight rotation
            
            else:
                # If nothing is found, slowly yaw to scan the room
                robot.yaw(mag=0.2, time=0.1)

    except KeyboardInterrupt:
        robot.stop(time=0)
        print("Manual Override: Stopping.")
