# required for motor movement
import serial
import time
from motorAPI import Movement

"""
Objective: test out the motor movement to ensure that the code corresponds with hardware
surge moves the robot forward
yaw turns the robot in place
"""

monkey_bot = Movement()

try:
    monkey_bot.surge(0.3,3) #forward
    monkey_bot.stop(1)
    monkey_bot.yaw(-0.3,3)   #rotate
    monkey_bot.stop(1)      

    monkey_bot.surge(0.3,3) #forward
    monkey_bot.stop(1)
    monkey_bot.yaw(-0.3,5)   #rotate
    monkey_bot.stop(1)

    monkey_bot.surge(0.3,3) #forward
    monkey_bot.stop(1)
    monkey_bot.yaw(-0.3,5)   #rotate
    monkey_bot.stop(1)

    monkey_bot.surge(0.3,3) #forward
    monkey_bot.stop(1)
except KeyboardInterrupt:
    print("Script Cancelled")
finally: 
    monkey_bot.stop()
    monkey_bot.close()
