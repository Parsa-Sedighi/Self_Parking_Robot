import depthai as dai
import numpy as np
from monkey_bot.motorAPI import Movement # Importing your provided class
import time

# 1. Initialize Motors
robot = Movement()

# 2. Pipeline Setup for OAK-D
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
nn = pipeline.create(dai.node.NeuralNetwork)
xout = pipeline.create(dai.node.XLinkOut)

xout.setStreamName("detections")
cam.setPreviewSize(28, 28)
cam.setInterleaved(False)
nn.setBlobPath("computer_vision/digit_model.blob") # Use your new blob

cam.preview.link(nn.input)
nn.out.link(xout.input)

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