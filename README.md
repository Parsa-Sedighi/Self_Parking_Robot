# SelfParkingRobot
Repository containing our codebase for the project.

# Create the virtual environment
py -3.11 -m venv venv

# Activate it
Linux/MacOS:
source venv/bin/activate 
Windows: 
.\venv\Scripts\Activate.ps1

# Install all dependencies at once
pip install -r requirements.txt

# Verify everything has been installed
python -c "import tensorflow as tf; import openvino; import depthai; import cv2; print(f'TF: {tf.__version__} | OV: {openvino.__version__} | DepthAI: {depthai.__version__} | OpenCV: {cv2.__version__}')"

# Should see the following on terminal:
TF: 2.15.0 | OV: 2024.0.0-14509-34caeefd078-releases/2024/0 | DepthAI: 3.3.0 | OpenCV: 4.13.0



Link to official MonkeyBot GitHub:
https://github.com/colin-szeto/monkey_bot/tree/main 


# First time set up on Raspberri Pi

1. Generate SSH key on the Pi:

```ssh-keygen -t ed25519 -C "your_email@example.com"```


2. Copy the public key:

```cat ~/.ssh/id_ed25519.pub```

3. Add it to GitHub → Settings → SSH and GPG keys → New SSH key.

4. Change your remote URL to SSH:

```git remote set-url origin git@github.com:colin-szeto/monkey_bot.git```


### Detecting the usb gamepad
1. List all input devices
ls -l /dev/input/by-id/


On a Pi with an F310 plugged in, you should see something like:

usb-Logitech_Gamepad_F310-event-joystick -> ../event3



