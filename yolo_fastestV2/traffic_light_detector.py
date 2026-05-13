"""
Traffic Light Detection Module
==============================

Detects traffic light colors (Red, Yellow, Green) and sends appropriate control commands.
All parameters are imported from constants.py for easy adjustment.

Author: Auto Vehicle Control System
Date: 2026-05-13
"""

import os
import cv2
import requests
import torch
import numpy as np
import model.detector
import utils.utils
from constants import (
    CONFIG_FILE,
    WEIGHTS_PATH,
    DEVICE,
    CONTROL_URL,
    NMS_CONF_THRESHOLD,
    NMS_IOU_THRESHOLD,
    TRAFFIC_LIGHT_LOWER_RED1,
    TRAFFIC_LIGHT_UPPER_RED1,
    TRAFFIC_LIGHT_LOWER_RED2,
    TRAFFIC_LIGHT_UPPER_RED2,
    TRAFFIC_LIGHT_LOWER_GREEN,
    TRAFFIC_LIGHT_UPPER_GREEN,
    TRAFFIC_LIGHT_LOWER_YELLOW,
    TRAFFIC_LIGHT_UPPER_YELLOW,
    TRAFFIC_LIGHT_RED_RATIO_THRESHOLD,
    TRAFFIC_LIGHT_GREEN_RATIO_THRESHOLD,
    TRAFFIC_LIGHT_YELLOW_RATIO_THRESHOLD,
    TRAFFIC_LIGHT_CATEGORIES,
    COMMAND_STOP,
    COMMAND_CONTINUE,
    DEBUG_MODE,
)


class TrafficLightDetector:
    """Traffic light detection and control module"""
    
    def __init__(self, control_url, state_lock):
        self.control_url = control_url
        self.state_lock = state_lock
        
        self.cfg = utils.utils.load_datafile(CONFIG_FILE)
        self.weights = WEIGHTS_PATH
        assert os.path.exists(self.weights), f"❌ Please specify correct model path: {self.weights}"
        
        self.device = DEVICE
        self.model = model.detector.Detector(self.cfg["classes"], self.cfg["anchor_num"], True).to(self.device)
        self.model.load_state_dict(torch.load(self.weights, map_location=self.device))
        self.model.eval()
        
        # Red color range (HSV) - 红色分为两个范围
        self.lower_red1 = TRAFFIC_LIGHT_LOWER_RED1
        self.upper_red1 = TRAFFIC_LIGHT_UPPER_RED1
        self.lower_red2 = TRAFFIC_LIGHT_LOWER_RED2
        self.upper_red2 = TRAFFIC_LIGHT_UPPER_RED2
        
        # Green color range (HSV)
        self.lower_green = TRAFFIC_LIGHT_LOWER_GREEN
        self.upper_green = TRAFFIC_LIGHT_UPPER_GREEN
        
        # Yellow color range (HSV)
        self.lower_yellow = TRAFFIC_LIGHT_LOWER_YELLOW
        self.upper_yellow = TRAFFIC_LIGHT_UPPER_YELLOW
        
        # Color detection thresholds
        self.red_ratio_threshold = TRAFFIC_LIGHT_RED_RATIO_THRESHOLD
        self.green_ratio_threshold = TRAFFIC_LIGHT_GREEN_RATIO_THRESHOLD
        self.yellow_ratio_threshold = TRAFFIC_LIGHT_YELLOW_RATIO_THRESHOLD
        
        # State tracking
        self.last_light_state = None
        self.red_light_detected = False
        self.traffic_light_state = None
        self.light_state_changed = False
        
        # Statistics
        self.frame_count = 0
        self.traffic_lights_detected = 0
    
    def detect_light_color(self, roi):
        """
        Detect traffic light color in ROI region
        
        Priority: RED > YELLOW > GREEN
        
        Args:
            roi: Region of interest (cropped image around traffic light)
            
        Returns:
            str: "RED", "YELLOW", "GREEN", or None if no color detected
        """
        if roi is None or roi.size == 0:
            return None
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Detect red color (two ranges due to HSV wraparound)
        mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        red_count = cv2.countNonZero(mask_red)
        
        # Detect yellow color
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        yellow_count = cv2.countNonZero(mask_yellow)
        
        # Detect green color
        mask_green = cv2.inRange(hsv, self.lower_green, self.upper_green)
        green_count = cv2.countNonZero(mask_green)
        
        total_pixels = roi.shape[0] * roi.shape[1]
        red_ratio = red_count / total_pixels
        yellow_ratio = yellow_count / total_pixels
        green_ratio = green_count / total_pixels
        
        if DEBUG_MODE:
            print(f"[TL COLOR] Red: {red_ratio:.3f} | Yellow: {yellow_ratio:.3f} | Green: {green_ratio:.3f}")
        
        # Priority: RED > YELLOW > GREEN
        if red_ratio > self.red_ratio_threshold:
            return "RED"
        elif yellow_ratio > self.yellow_ratio_threshold:
            return "YELLOW"
        elif green_ratio > self.green_ratio_threshold:
            return "GREEN"
        else:
            return None
    
    def extract_roi_from_box(self, frame, box, scale_h, scale_w):
        """
        Extract region of interest from detection box
        
        Args:
            frame: Input frame
            box: Detection box coordinates
            scale_h: Height scale factor
            scale_w: Width scale factor
            
        Returns:
            tuple: (roi image, coordinates tuple)
        """
        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        roi = frame[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)
    
    def is_traffic_light(self, category):
        """Check if detected object is traffic light"""
        return category.lower() in TRAFFIC_LIGHT_CATEGORIES
    
    def detect_state_change(self, current_color):
        """
        Detect if traffic light state changed from RED to YELLOW or YELLOW to GREEN
        
        Args:
            current_color: Current detected color
            
        Returns:
            bool: True if state changed, False otherwise
        """
        old_state = self.traffic_light_state
        self.traffic_light_state = current_color
        
        if old_state == "RED" and current_color in ["YELLOW", "GREEN"]:
            self.light_state_changed = True
            return True
        elif old_state == "YELLOW" and current_color == "GREEN":
            self.light_state_changed = True
            return True
        
        self.light_state_changed = False
        return False
    
    def get_command_by_light_color(self, light_color):
        """
        Determine movement command based on traffic light color
        
        Args:
            light_color: Detected light color
            
        Returns:
            tuple: (command, status_string)
        """
        if light_color == "RED":
            command = COMMAND_STOP
            status = "🔴 RED LIGHT - STOP"
        elif light_color == "YELLOW":
            command = COMMAND_STOP  # Yellow means prepare to stop
            status = "🟡 YELLOW LIGHT - CAUTION"
        elif light_color == "GREEN":
            command = COMMAND_CONTINUE
            status = "🟢 GREEN LIGHT - GO"
        else:
            command = COMMAND_STOP
            status = "⚪ UNKNOWN LIGHT - STOP"
        
        return command, status
    
    def send_command(self, command):
        """
        Send movement command to vehicle
        
        Args:
            command: Command string (STOP, CONTINUE, etc.)
            
        Returns:
            bool: True if command sent successfully
        """
        try:
            response = requests.post(self.control_url, json={'command': command}, timeout=2)
            return response.status_code == 200
        except Exception as e:
            print(f"[TL ERROR] Failed to send command: {str(e)}")
            return False
    
    def should_execute(self, light_color):
        """Check if traffic light command should be executed"""
        return light_color is not None
    
    def run(self, cap, LABEL_NAMES):
        """
        Main detection loop for traffic light monitoring
        
        Args:
            cap: Video capture object
            LABEL_NAMES: List of class names
        """
        print("[TRAFFIC LIGHT] Starting traffic light detection thread...")
        print(f"[TRAFFIC LIGHT] Red threshold: {self.red_ratio_threshold:.2%}")
        print(f"[TRAFFIC LIGHT] Yellow threshold: {self.yellow_ratio_threshold:.2%}")
        print(f"[TRAFFIC LIGHT] Green threshold: {self.green_ratio_threshold:.2%}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[TRAFFIC LIGHT] Failed to read frame - exiting")
                break
            
            self.frame_count += 1
            
            # Prepare image for YOLO inference
            res_img = cv2.resize(frame, (self.cfg["width"], self.cfg["height"]), 
                               interpolation=cv2.INTER_LINEAR)
            img = res_img.reshape(1, self.cfg["height"], self.cfg["width"], 3)
            img = torch.from_numpy(img.transpose(0, 3, 1, 2))
            img = img.to(self.device).float() / 255.0
            
            # Run YOLO inference
            preds = self.model(img)
            output = utils.utils.handel_preds(preds, self.cfg, self.device)
            output_boxes = utils.utils.non_max_suppression(
                output, 
                conf_thres=NMS_CONF_THRESHOLD, 
                iou_thres=NMS_IOU_THRESHOLD
            )
            
            h, w, _ = frame.shape
            scale_h, scale_w = h / self.cfg["height"], w / self.cfg["width"]
            
            # If no objects detected, stop
            if len(output_boxes[0]) == 0:
                self.send_command(COMMAND_STOP)
                if DEBUG_MODE and self.frame_count % 30 == 0:
                    print("[TRAFFIC LIGHT] No objects detected - Stop")
            
            # Process detections
            for box in output_boxes[0]:
                box = box.tolist()
                
                obj_score = box[4]
                category = LABEL_NAMES[int(box[5])]
                
                x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
                x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)
                
                if self.is_traffic_light(category):
                    self.traffic_lights_detected += 1
                    roi, coords = self.extract_roi_from_box(frame, box, scale_h, scale_w)
                    light_color = self.detect_light_color(roi)
                    
                    if light_color:
                        state_changed = self.detect_state_change(light_color)
                        
                        # Update shared state
                        with self.state_lock:
                            if light_color == "RED":
                                self.red_light_detected = True
                            else:
                                self.red_light_detected = False
                        
                        # Send control command
                        command, status = self.get_command_by_light_color(light_color)
                        self.send_command(command)
                        
                        if DEBUG_MODE:
                            print(f"[TRAFFIC LIGHT] {status} | Confidence: {obj_score:.2f}")
                        
                        if state_changed:
                            print(f"[TRAFFIC LIGHT] ⚡ State changed: {self.last_light_state} → {light_color}")
                        
                        self.last_light_state = light_color
                        
                        # Draw detection box with color
                        color_map = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0)}
                        box_color = color_map.get(light_color, (255, 255, 0))
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                        cv2.putText(frame, f"{category}: {light_color}", (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(frame, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
                
                cv2.putText(frame, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Traffic Light Detection', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def get_state(self):
        """Get current traffic light state"""
        return {
            'red_light_detected': self.red_light_detected,
            'traffic_light_state': self.traffic_light_state,
            'light_state_changed': self.light_state_changed,
            'frame_count': self.frame_count,
            'lights_detected': self.traffic_lights_detected
        }
