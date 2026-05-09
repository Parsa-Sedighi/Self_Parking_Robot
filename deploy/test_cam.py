import depthai as dai
import cv2

pipeline = dai.Pipeline()
cam = pipeline.createColorCamera()
cam.setPreviewSize(640, 480)

xout = pipeline.createXLinkOut()
xout.setStreamName("video")
cam.preview.link(xout.input)

with dai.Device(pipeline) as device:
    q = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    print("Camera connected. Waiting for first frame...")
    img = q.get() # If it hangs here, it's a hardware/USB problem.
    print("Success! Camera is streaming.")
