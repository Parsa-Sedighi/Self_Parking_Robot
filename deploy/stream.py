import depthai as dai
import cv2

pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)

# --- THE FIX: LOW RES + LOW FPS ---
# We use 720p sensor res but only a 300x300 preview to save RAM
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
cam.setPreviewSize(300, 300) 
cam.setFps(10)  # Lowering to 10 FPS reduces the load by 66%
cam.setInterleaved(False)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam.preview.link(xout.input)

with dai.Device(pipeline) as device:
    # Force USB 2.0 mode if the cable or Pi port is struggling
    # print(f"USB Speed: {device.getUsbSpeed()}") 
    
    q = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    
    print("Ultralight Feed Active. Press 'q' to quit.")
    
    while True:
        in_frame = q.tryGet()
        if in_frame is not None:
            frame = in_frame.getCvFrame()
            cv2.imshow("ULTRALIGHT VIEW", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
