import depthai as dai
import cv2

# 1. Create the Pipeline
pipeline = dai.Pipeline()

# 2. Define the Color Camera source
cam_rgb = pipeline.createColorCamera() # Alternative creation method
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(30)

# 3. Create an output "XLink" 
# We use the explicit node class here to avoid 'AttributeError'
xout_rgb = pipeline.create(dai.node.XLinkOut) 
xout_rgb.setStreamName("video")

# 4. Link the camera output to the PC output
cam_rgb.preview.link(xout_rgb.input)

# 5. Start the Device and the Pipeline
try:
    # This searches for the OAK-D on your USB ports
    with dai.Device(pipeline) as device:
        # Get the output queue
        q_rgb = device.getOutputQueue(name="video", maxSize=4, blocking=False)

        print("OAK-D Live Feed Started. Press 'q' to exit.")

        while True:
            # Pull the data from the queue
            in_rgb = q_rgb.get() 
            frame = in_rgb.getCvFrame()

            # Display the feed
            cv2.imshow("Robot View - OAK-D", frame)

            # Break loop on 'q'
            if cv2.waitKey(1) == ord('q'):
                break
except Exception as e:
    print(f"\n--- Connection Error ---")
    print(f"Details: {e}")
    print("Tip: Ensure the OAK-D is plugged into a USB 3.0 port with a high-quality cable.")
finally:
    cv2.destroyAllWindows()