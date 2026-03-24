import serial
import time
"""
Objective: 
the interface to the ardiuno
ensures the commands are mixed correctly for less cognitive load on the user
"""

class Movement():
    def __init__(self):
        # --- Setup ---
        self.arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1) # ttyUSB0 may need to be changed to ttyACM0 use "sudo dmesg --follow" to determine the port of the ardiuno
        time.sleep(2)  # let Arduino reset

    def send_to_arduino(self,m1, m2, delay=0):
        # def send_to_arduino(self,m1, m2, delay=0):
        # objective: Send motor values, optionally wait for some time.
        # m1 = value sent in serial monitor for motor 1
        # m2 = value sent in serial monitor for motor 2
        # delay = default value 0, duration before allowing next command to execute
        cmd = f"{m1},{m2}\n" # this is the format sent the ardiuno
        self.arduino.write(cmd.encode())
        #print(f"Sent: {cmd.strip()}")
        if delay != False:
            if delay > 0:
                time.sleep(delay)
        

    def surge(self,mag,time):
        # def surge(self,mag,time):
        # objective: move the robot forward
        # mag = mangitude
        # time = duration of command
        print("surge")
        self.send_to_arduino(255*mag, 255*mag, delay=time)

    def yaw(self,mag,time):
        # def yaw(self,mag,time):
        # objective: spin the robot in place
        # mag = mangitude (positive magniture is clockwise rotation)
        # time = duration of command
        print("yaw")
        self.send_to_arduino(-255*mag,255*mag, delay=time)

    def stop(self,time):
        # def stop(self,time)
        # objective: send command to ardiuno to stop the motors
        # time = duration of command
        print("stop")
        self.send_to_arduino(0,0, delay=time)

    def close(self):
        arduino.close()
        print("Serial closed.")
