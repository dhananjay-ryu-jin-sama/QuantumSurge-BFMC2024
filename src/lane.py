lane.py

import time 
import cv2
import numpy as np
from picamera2 import Picamera2

from src.hardware.serialhandler.threads.messageconverter import MessageConverter
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import SpeedMotor, SteerMotor
from src.utils.messages.messageHandlerSender import messageHandlerSender

class LaneFollowingSystem(ThreadWithStop):
    def _init_(self, queues, serialCom, logFile, logger, debugger=False):
        super(LaneFollowingSystem, self)._init_()
        self.queuesList = queues
        self.serialCom = serialCom
        self.logFile = logFile
        self.logger = logger
        self.debugger = debugger

        self.running = False
        self.messageConverter = MessageConverter()
        self.speedMotorSender = messageHandlerSender(self.queuesList, SpeedMotor)
        self.steerMotorSender = messageHandlerSender(self.queuesList, SteerMotor)

        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = (640, 480)
        self.picam2.preview_configuration.main.format = "RGB888"
        self.picam2.configure("preview")
        self.picam2.start()

        self.constant_speed = 100  # Maintain a constant speed
        self.Kp = 3.0  # Increased proportional control for faster correction
        self.Kd =  1.2# Added derivative control for sharper turns
        self.last_offset = 0

    def sendToSerial(self, msg):
        command_msg = self.messageConverter.get_command(**msg)
        if command_msg != "error":
            self.serialCom.write(command_msg.encode("ascii"))
            self.logFile.write(command_msg)
            if self.debugger:
                self.logger.info(f"Sent to serial: {command_msg}")

    def set_motor_speed(self, speed_value):
        command = {"action": "speed", "speed": speed_value}
        self.sendToSerial(command)
        if self.debugger:
            self.logger.info(f"Motor speed set to: {speed_value}")

    def set_steering_angle(self, angle):
        command = {"action": "steer", "steerAngle": angle}
        self.sendToSerial(command)
        if self.debugger:
            self.logger.info(f"Steering angle set to: {angle}")

    def set_kl_mode(self, mode):
        command = {"action": "kl", "mode": mode}
        self.sendToSerial(command)
        self.logger.info(f"KL mode set to: {mode}")

    def filter_white_lanes(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)
        _, white_mask = cv2.threshold(l_channel, 200, 255, cv2.THRESH_BINARY)
        height, width = white_mask.shape
        mask = np.zeros_like(white_mask)
        mask[height//2:, :] = 255
        return cv2.bitwise_and(white_mask, mask)

    def find_nearest_white_points(self, binary_image, mid_x, mid_y):
        height, width = binary_image.shape
        left_x, right_x = None, None
        for x in range(mid_x, 0, -1):
            if binary_image[mid_y, x] == 255:
                left_x = x
                break
        for x in range(mid_x, width):
            if binary_image[mid_y, x] == 255:
                right_x = x
                break
        return left_x, right_x

    def run(self):
        self.running = True
        self.set_kl_mode(30)
        time.sleep(1)
        self.set_motor_speed(self.constant_speed)

        while self._running:
            self.set_motor_speed(self.constant_speed)
            frame = self.picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            white_lane_mask = self.filter_white_lanes(frame)

            height, width = white_lane_mask.shape
            mid_x = width // 2
            mid_y = height * 2 // 3  # Adjusted mid_y for better positioning

            left_x, right_x = self.find_nearest_white_points(white_lane_mask, mid_x, mid_y)

            if left_x is not None and right_x is not None:
                lane_midpoint = (left_x + right_x) // 2
                offset = lane_midpoint - mid_x
                derivative = offset - self.last_offset
                steering_angle = np.clip(self.Kp * offset + self.Kd * derivative, -230, 230)
                self.set_steering_angle(int(steering_angle))
                self.last_offset = offset
            elif left_x is not None:
                self.set_steering_angle(229)  # Turn full right
            elif right_x is not None:
                self.set_steering_angle(-229)  # Turn full left

            cv2.circle(frame, (mid_x, mid_y), 5, (0, 255, 0), -1)  # Center point (Green)
            if left_x is not None:
                cv2.circle(frame, (left_x, mid_y), 5, (255, 0, 0), -1)  # Left lane detected (Blue)
            if right_x is not None:
                cv2.circle(frame, (right_x, mid_y), 5, (0, 0, 255), -1)  # Right lane detected (Red)
            if left_x is not None and right_x is not None:
                lane_midpoint = (left_x + right_x) // 2
                cv2.circle(frame, (lane_midpoint, mid_y), 5, (0, 255, 255), -1)  # Midpoint (Yellow)
                cv2.circle(frame, (lane_midpoint, mid_y), 5, (0, 255, 0), -1)  # Ensure Green coincides with Yellow

            cv2.imshow("Lane Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        cv2.destroyAllWindows()
        self.picam2.stop()

    def stop(self):
        self.set_motor_speed(0)
        super(LaneFollowingSystem, self).stop()
        self.logger.info("Vehicle stopped and thread terminated.")

if _name_ == "_main_":
    from multiprocessing import Queue
    import serial
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("LaneFollowingSystem")

    devFile = "/dev/ttyACM0"
    serialCom = serial.Serial(devFile, 115200, timeout=0.1)
    serialCom.flushInput()
    serialCom.flushOutput()

    queueList = {
        "Critical": Queue(),
        "Warning": Queue(),
        "General": Queue(),
        "Config": Queue(),
    }

    logFile = open("historyFile.txt", "w")

    lane_follow_thread = LaneFollowingSystem(queueList, serialCom, logFile, logger, debugger=True)
    lane_follow_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        lane_follow_thread.stop()