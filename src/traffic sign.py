
import json
import time
import cv2
import numpy as np
import torch
from picamera2 import Picamera2
from ultralytics import YOLO

from src.hardware.serialhandler.threads.messageconverter import MessageConverter
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import SpeedMotor, SteerMotor
from src.utils.messages.messageHandlerSender import messageHandlerSender


class AutonomousDrivingSystem(ThreadWithStop):
    def _init_(self, queues, serialCom, logFile, logger, debugger=False):
        super(AutonomousDrivingSystem, self)._init_()
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

        # Initialize PiCamera
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = (640, 480)
        self.picam2.preview_configuration.main.format = "RGB888"
        self.picam2.configure("preview")
        self.picam2.start()

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
            self.logger.info("Resuming constant speed.")
            self.set_motor_speed(self.constant_speed)

        elif 'traffic sign - crosswalk' in detected_signs:
            self.logger.info("Crosswalk detected. Slowing down for 3 seconds.")
            self.set_motor_speed(int(self.constant_speed * 0.4))
            time.sleep(3)
            self.logger.info("Regaining initial speed.")
            self.set_motor_speed(self.constant_speed)

        elif 'traffic sign - highway entrance' in detected_signs:
            self.logger.info("Highway entrance detected. Speeding up.")
            self.set_motor_speed(400)

        elif 'traffic sign - highway exit' in detected_signs:
            self.logger.info("Highway exit detected. Slowing down.")
            self.set_motor_speed(300)

        elif 'traffic sign - no entry' in detected_signs:
            self.logger.info("No-entry detected. Stopping.")
            self.set_motor_speed(0)
            time.sleep(2)

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

    def run(self):
        self.running = True
        self.set_kl_mode(30)
        time.sleep(1)
        self.set_motor_speed(self.constant_speed)

        while self._running:
            frame = self.picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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
        self.picam2.stop()

    def stop(self):
        self.set_motor_speed(0)
        super(AutonomousDrivingSystem, self).stop()
        self.logger.info("Vehicle stopped and thread terminated.")


if _name_ == "_main_":
    from multiprocessing import Queue
    import serial
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("AutonomousDrivingSystem")

    devFile = "/dev/ttyACM0"
    serialCom = serial.Serial(devFile, 115200, timeout=0.1)

    queueList = {"Critical": Queue(), "Warning": Queue(), "General": Queue(), "Config": Queue()}
    logFile = open("historyFile.txt", "w")

    autonomous_system = AutonomousDrivingSystem(queueList, serialCom, logFile, logger, debugger=True)
    autonomous_system.start()