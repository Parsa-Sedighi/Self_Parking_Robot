import cv2
import numpy as np

# required for motor movement
import serial
import time
from motorAPI import Movement

cap = cv2.VideoCapture(0)
monkey_bot = Movement()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    center_line_x = width // 2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.rotate(hsv, cv2.ROTATE_180) # this line is required if your camera is mounted upside down

    # Red HSV ranges
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    # Blue HSV range
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 194 , 188])

    # Masks
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_x = blue_x = None

    if contours_red:
        largest_red = max(contours_red, key=cv2.contourArea)
        M_red = cv2.moments(largest_red)
        if M_red["m00"] != 0:
            red_x = int(M_red["m10"] / M_red["m00"])
            cv2.drawContours(frame, [largest_red], -1, (0, 0, 255), 2)

    if contours_blue:
        largest_blue = max(contours_blue, key=cv2.contourArea)
        M_blue = cv2.moments(largest_blue)
        if M_blue["m00"] != 0:
            blue_x = int(M_blue["m10"] / M_blue["m00"])
            cv2.drawContours(frame, [largest_blue], -1, (255, 0, 0), 2)

    # Draw vertical center line
    cv2.line(frame, (center_line_x, 0), (center_line_x, height), (0, 255, 255), 2)

    # Draw midpoint and calculate difference
    if red_x is not None and blue_x is not None:
        midpoint_x = (red_x + blue_x) // 2
        midpoint_y = height // 2
        cv2.circle(frame, (midpoint_x, midpoint_y), 8, (0, 255, 0), -1)
        cv2.putText(frame, f'Mid X: {midpoint_x}', (midpoint_x + 10, midpoint_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Calculate and print difference
        diff = (midpoint_x - center_line_x)/width
        print(f"Midpoint X: {midpoint_x}, Center Line X: {center_line_x}, Difference: {diff}")
    
        # diff = (midpoint_x - center_line_x)/width
        # this is the logic of attempting to keep the midpoint between the red and blue blocks at the center of the frame
        if midpoint_x < width/5:
            monkey_bot.yaw(0.25,False)
        elif midpoint_x > width * 4/5:
            monkey_bot.yaw(-0.25,False)
        else: 
            monkey_bot.surge(0.25,False)
    else: 
        if red_x:
            monkey_bot.yaw(0.15,False)
        else: 
            if blue_x:
                monkey_bot.yaw(-0.15,False)


    if False:
        cv2.imshow("Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
