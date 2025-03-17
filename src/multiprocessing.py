import json
import time
import cv2
import numpy as np
import torch
from picamera2 import Picamera2
from ultralytics import YOLO
from multiprocessing import Queue, Process
import serial
import logging
from threading import Thread, Event

from src.hardware.serialhandler.threads.messageconverter import MessageConverter
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import SpeedMotor, SteerMotor
from src.utils.messages.messageHandlerSender import messageHandlerSender


class TrafficSignDetection(ThreadWithStop):
    def _init_(self, queues, serialCom, logFile, logger, debugger=False):
        super(TrafficSignDetection, self)._init_()
        self.queuesList = queues
        self.serialCom = serialCom
        self.logFile = logFile
        self.logger = logger
        self.debugger = debugger

        self.running = False
        self.messageConverter = MessageConverter()
        self.speedMotorSender = messageHandlerSender(self.queuesList, SpeedMotor)
        self.steerMotorSender = messageHandlerSender(self.queuesList, SteerMotor)

        # Load YOLOv8 model for traffic sign detection
        self.model = YOLO('/home/raspi/Downloads/traffic_sign_final.pt')
        self.class_names = self.model.names

        # Default speed
        self.constant_speed = 100

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

    def process_detections(self, detected_signs):
        """Modify vehicle behavior based on detected traffic signs."""
        if 'traffic sign - stop' in detected_signs:
            self.logger.info("Stop sign detected. Stopping for 3 seconds.")
            self.set_motor_speed(0)
            time.sleep(3)
            self.logger.info("Making a slight left turn.")
            self.set_steering_angle(230)# Slight left turn
            time.sleep(4)
            self.set_steering_angle(0)  # Straighten
            self.logger.info("Resuming constant speed.")
            self.set_motor_speed(self.constant_speed)

        elif 'traffic sign - crosswalk' in detected_signs:
            self.logger.info("Crosswalk detected. Slowing down for 3 seconds.")
            self.set_motor_speed(int(self.constant_speed * 0.4))
            time.sleep(5)
            self.logger.info("Regaining initial speed.")
            self.set_motor_speed(self.constant_speed)

        elif 'traffic sign - highway entrance' in detected_signs:
            self.logger.info("Highway entrance detected. Speeding up.")
            self.set_motor_speed(200)

        elif 'traffic sign - highway exit' in detected_signs:
            self.logger.info("Highway exit detected. Slowing down.")
            self.set_motor_speed(100)

        elif 'traffic sign - no entry' in detected_signs:
            self.logger.info("No-entry detected. Turning left.")
            self.set_motor_speed(80)  # Slow down for the turn
            self.set_steering_angle(230)  # Turn left
            time.sleep(8)
            self.set_steering_angle(0)  # Straighten
            self.set_motor_speed(self.constant_speed)

        elif 'traffic sign - priority' in detected_signs:
            self.logger.info("Priority detected. Turning right.")
            self.set_motor_speed(50)  # Slow down for the turn
            self.set_steering_angle(-230)  # Turn right
            time.sleep(4)
            self.set_steering_angle(0)  # Straighten
            self.set_motor_speed(self.constant_speed)

        elif 'traffic sign - roundabout' in detected_signs:
            self.logger.info("Roundabout detected. Slowing down.")
            self.set_motor_speed(int(self.constant_speed * 0.6))
            self.set_steering_angle(230)
            time.sleep(0.5)
            self.set_steering_angle(-230)

        elif 'pedestrian on road' in detected_signs:
            self.logger.info("Pedestrian detected! Stopping the vehicle until clear.")
            self.set_motor_speed(0)

            # Wait until pedestrian disappears
            while True:
                frame = self.picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                results = self.model(frame_bgr)

                pedestrian_still_detected = False
                for result in results[0].boxes.data:
                    confidence, class_id = result[4], int(result[5])
                    class_name = self.class_names[class_id].lower()
                    if confidence > 0.5 and class_name == 'pedestrian on road':
                        pedestrian_still_detected = True
                        break

                if not pedestrian_still_detected:
                    self.logger.info("Pedestrian cleared! Resuming normal speed.")
                    self.set_motor_speed(self.constant_speed)
                    break

    def run(self, frame_queue):
        self.running = True
        self.set_kl_mode(30)
        time.sleep(1)
        self.set_motor_speed(self.constant_speed)

        while self._running:
            frame = frame_queue.get()
            frame_bgr = frame  # Use the original frame without color space conversion

            # Traffic sign detection processing
            results = self.model(frame_bgr)
            detected_signs = []
            for result in results[0].boxes.data:
                x1, y1, x2, y2, confidence, class_id = result.tolist()
                if confidence > 0.5:
                    class_name = self.class_names[int(class_id)]
                    detected_signs.append(class_name.lower())
                    cv2.rectangle(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame_bgr, f'{class_name} ({confidence:.2f})',
                                (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            self.process_detections(detected_signs)

            cv2.imshow("Traffic Sign Detection", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        cv2.destroyAllWindows()

    def stop(self):
        self.set_motor_speed(0)
        super(TrafficSignDetection, self).stop()
        self.logger.info("Vehicle stopped and thread terminated.")


class LaneFollowing(ThreadWithStop):
    def _init_(self, queues, serialCom, logFile, logger, debugger=False):
        super(LaneFollowing, self)._init_()
        self.queuesList = queues
        self.serialCom = serialCom
        self.logFile = logFile
        self.logger = logger
        self.debugger = debugger

        self.running = False
        self.messageConverter = MessageConverter()
        self.speedMotorSender = messageHandlerSender(self.queuesList, SpeedMotor)
        self.steerMotorSender = messageHandlerSender(self.queuesList, SteerMotor)

        self.constant_speed = 100  # Maintain a constant speed
        self.Kp = 3.0  # Increased proportional control for faster correction
        self.Kd =  1.2  # Added derivative control for sharper turns
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

    def run(self, frame_queue):
        self.running = True
        self.set_kl_mode(30)
        time.sleep(1)
        self.set_motor_speed(self.constant_speed)

        while self._running:
            frame = frame_queue.get()
            frame_bgr = frame  # Use the original frame without color space conversion

            white_lane_mask = self.filter_white_lanes(frame_bgr)

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

            cv2.circle(frame_bgr, (mid_x, mid_y), 5, (0, 255, 0), -1)  # Center point (Green)
            if left_x is not None:
                cv2.circle(frame_bgr, (left_x, mid_y), 5, (255, 0, 0), -1)  # Left lane detected (Blue)
            if right_x is not None:
                cv2.circle(frame_bgr, (right_x, mid_y), 5, (0, 0, 255), -1)  # Right lane detected (Red)
            if left_x is not None and right_x is not None:
                lane_midpoint = (left_x + right_x) // 2
                cv2.circle(frame_bgr, (lane_midpoint, mid_y), 5, (0, 255, 255), -1)  # Midpoint (Yellow)
                cv2.circle(frame_bgr, (lane_midpoint, mid_y), 5, (0, 255, 0), -1)  # Ensure Green coincides with Yellow

            cv2.imshow("Lane Detection", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        cv2.destroyAllWindows()

    def stop(self):
        self.set_motor_speed(0)
        super(LaneFollowing, self).stop()
        self.logger.info("Vehicle stopped and thread terminated.")


def frame_capture(picam2, frame_queue, stop_event):
    while not stop_event.is_set():
        if frame_queue.qsize() < 10:  # Limit the number of frames in the queue
            frame = picam2.capture_array()
            frame_queue.put(frame)
        time.sleep(0.01)  # Adjust the sleep time as needed

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("AutonomousCarSystem")

    devFile = "/dev/ttyACM0"
    serialCom = serial.Serial(devFile, 115200, timeout=0.1)

    queueList = {"Critical": Queue(), "Warning": Queue(), "General": Queue(), "Config": Queue()}
    logFile = open("historyFile.txt", "w")

    frame_queue = Queue()
    stop_event = Event()

    traffic_sign_detection = TrafficSignDetection(queueList, serialCom, logFile, logger, debugger=True)
    lane_following = LaneFollowing(queueList, serialCom, logFile, logger, debugger=True)

    traffic_sign_process = Process(target=traffic_sign_detection.run, args=(frame_queue,))
    lane_following_process = Process(target=lane_following.run, args=(frame_queue,))

    traffic_sign_process.start()
    lane_following_process.start()

    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (320, 240)  # Reduce resolution to 320x240
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()

    frame_capture_thread = Thread(target=frame_capture, args=(picam2, frame_queue, stop_event))
    frame_capture_thread.start()

    try:
        while True:
            time.sleep(0.1)  # Main thread can perform other tasks if needed
    except KeyboardInterrupt:
        stop_event.set()
        frame_capture_thread.join()
        traffic_sign_detection.stop()
        lane_following.stop()
        traffic_sign_process.join()
        lane_following_process.join()
        picam2.stop()
        logFile.close()

if _name_ == "_main_":
    main()

