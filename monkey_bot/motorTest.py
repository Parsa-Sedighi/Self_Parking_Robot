# required for motor movement
import serial
import time
from motorAPI import Movement

"""
Objective: test out the motor movement to ensure that the code corresponds with hardware
surge moves the robot forward
yaw turns the robot in place
"""
try:
    monkey_bot = Movement()
    monkey_bot.surge(0.25,3)
    monkey_bot.stop(1)
    monkey_bot.yaw(0.25,3)
    monkey_bot.stop(1)
except KeyboardInterrupt:
    print("Script Cancelled")
finally: 
    monkey_bot.stop()
    monkey_bot.close()
