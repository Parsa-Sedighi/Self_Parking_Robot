import depthai as dai
import cv2

# 1. Initialize the Pipeline
pipeline = dai.Pipeline()

# 2. Define the Camera Node
# Using ColorCamera as it is the most stable for your specific 2.32.0.0 build
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setPreviewSize(1280, 720) 

# 3. Define the XLinkOut Node
# In 2.32.0.0, this MUST be dai.node.XLinkOut (Correct Casing)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")

# 4. Link the Nodes
cam.preview.link(xout.input)

# 5. Device Lifecycle
with dai.Device(pipeline) as device:
    # Verify SuperSpeed (USB 3.0) on your Pi 4
    print(f"--- PI 4 SYSTEM ONLINE ---")
    print(f"DepthAI Version: {dai.__version__}")
    print(f"Bus Speed: {device.getUsbSpeed()}")
    
    q_video = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    print("Streaming... Press 'q' to quit.")
    while True:
        in_frame = q_video.get()
        frame = in_frame.getCvFrame()

        # Display on your HDMI monitor
        cv2.imshow("Pi 4 - OAK-D Stream (V2.32)", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
