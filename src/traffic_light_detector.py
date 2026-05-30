"""
Traffic Light Detection Module - SIMPLIFIED VERSION
====================================================

简化版红绿灯检测模块 - 使用直接的面积阈值
- 无状态平滑，无置信度检查
- 简洁高效，易于调试
- 检测逻辑：红灯面积 > 2000px 或 绿灯面积 > 2000px

Author: Auto Vehicle Control System
Date: 2026-05-26
"""

import cv2
import requests
import numpy as np
from threading import Lock
import time

from constants import (
    CONTROL_URL,
    TRAFFIC_LIGHT_LOWER_RED1,
    TRAFFIC_LIGHT_UPPER_RED1,
    TRAFFIC_LIGHT_LOWER_RED2,
    TRAFFIC_LIGHT_UPPER_RED2,
    TRAFFIC_LIGHT_LOWER_GREEN,
    TRAFFIC_LIGHT_UPPER_GREEN,
    COMMAND_STOP,
    COMMAND_CONTINUE,
    DEBUG_MODE,
)


class TrafficLightDetector:
    """
    Traffic light color detection - SIMPLIFIED VERSION
    
    简化的红绿灯检测器 - 使用直接的像素面积检测
    """
    
    def __init__(self, control_url, state_lock):
        """
        Initialize traffic light detector.
        
        Args:
            control_url: URL to send control commands to
            state_lock: Thread lock for shared state
        """
        self.control_url = control_url
        self.state_lock = state_lock
        
        # HSV Color Ranges
        self.lower_red1 = TRAFFIC_LIGHT_LOWER_RED1
        self.upper_red1 = TRAFFIC_LIGHT_UPPER_RED1
        self.lower_red2 = TRAFFIC_LIGHT_LOWER_RED2
        self.upper_red2 = TRAFFIC_LIGHT_UPPER_RED2
        
        self.lower_green = TRAFFIC_LIGHT_LOWER_GREEN
        self.upper_green = TRAFFIC_LIGHT_UPPER_GREEN
        
        # Area thresholds (in pixels)
        self.red_area_threshold = 2000      # 红灯面积阈值
        self.green_area_threshold = 2000    # 绿灯面积阈值
        
        # State tracking
        self.last_light_state = None
        self.red_light_detected = False
        self.traffic_light_state = None
        
        # Statistics
        self.frame_count = 0
        
        if DEBUG_MODE:
            print("\n[TL DETECTOR - SIMPLIFIED] Traffic Light Detector Initialized")
            print(f"[TL DETECTOR] Red threshold: {self.red_area_threshold} pixels")
            print(f"[TL DETECTOR] Green threshold: {self.green_area_threshold} pixels")
    
    def detect_light_color(self, frame):
        """
        Detect traffic light color in frame using direct area detection.
        
        Args:
            frame: Input frame
            
        Returns:
            tuple: (color_name, red_area, green_area)
        """
        if frame is None or frame.size == 0:
            return None, 0, 0
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect Red (two ranges)
        mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        red_area = cv2.countNonZero(mask_red)
        
        # Detect Green
        mask_green = cv2.inRange(hsv, self.lower_green, self.upper_green)
        green_area = cv2.countNonZero(mask_green)
        
        # Priority: RED > GREEN
        if red_area > self.red_area_threshold:
            return "RED", red_area, green_area
        elif green_area > self.green_area_threshold:
            return "GREEN", red_area, green_area
        else:
            return None, red_area, green_area
    
    def get_command_by_light_color(self, light_color):
        """
        Determine movement command based on traffic light color.
        
        Args:
            light_color: Detected light color
            
        Returns:
            tuple: (command, status_string)
        """
        if light_color == "RED":
            command = COMMAND_STOP
            status = "🔴 RED LIGHT - STOP"
        elif light_color == "GREEN":
            command = COMMAND_CONTINUE
            status = "🟢 GREEN LIGHT - GO"
        else:
            command = COMMAND_STOP
            status = "⚪ NO LIGHT DETECTED - STOP"
        
        return command, status
    
    def send_command(self, command):
        """
        Send movement command to vehicle.
        
        Args:
            command: Command string (STOP, CONTINUE, etc.)
            
        Returns:
            bool: True if command sent successfully
        """
        try:
            response = requests.post(self.control_url, json={'command': command}, timeout=2)
            return response.status_code == 200
        except Exception as e:
            print(f"[TL ERROR] Failed to send command '{command}': {str(e)}")
            return False
    
    def run(self, cap, LABEL_NAMES, exit_flag):
        """
        Main detection loop for traffic light monitoring.
        
        Args:
            cap: Video capture object
            LABEL_NAMES: List of class names (unused)
            exit_flag: Dict flag to signal thread exit {'flag': bool}
        """
        print("\n[TRAFFIC LIGHT] ===== SIMPLIFIED COLOR-BASED TRAFFIC LIGHT DETECTOR =====")
        print("[TRAFFIC LIGHT] Detection method: DIRECT AREA ANALYSIS (No state smoothing)")
        print(f"[TRAFFIC LIGHT] Red threshold: {self.red_area_threshold} pixels")
        print(f"[TRAFFIC LIGHT] Green threshold: {self.green_area_threshold} pixels")
        print("[TRAFFIC LIGHT] ========================================================================\n")
        
        last_sent_command = None
        command_debounce_time = 0
        
        while not exit_flag['flag']:
            ret, frame = cap.read()
            if not ret:
                if DEBUG_MODE:
                    print("[TRAFFIC LIGHT] Failed to read frame")
                break
            
            self.frame_count += 1
            
            # Detect light color
            light_color, red_area, green_area = self.detect_light_color(frame)
            
            # Update shared state
            with self.state_lock:
                if light_color == "RED":
                    self.red_light_detected = True
                else:
                    self.red_light_detected = False
            
            # Send control command (with debouncing)
            if light_color and (time.time() - command_debounce_time > 0.5 or 
                               light_color != self.last_light_state):
                command, status = self.get_command_by_light_color(light_color)
                
                self.send_command(command)
                command_debounce_time = time.time()
                
                if DEBUG_MODE or light_color != self.last_light_state:
                    print(f"[TRAFFIC LIGHT] {status} | Red={red_area} pixels | Green={green_area} pixels")
                
                self.last_light_state = light_color
                self.traffic_light_state = light_color
                last_sent_command = command
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag['flag'] = True
                break
        
        print("\n[TRAFFIC LIGHT] Detection thread stopped")
    
    def get_state(self):
        """Get current traffic light state for other modules."""
        return {
            'red_light_detected': self.red_light_detected,
            'traffic_light_state': self.traffic_light_state,
            'frame_count': self.frame_count,
        }
