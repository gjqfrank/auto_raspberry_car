"""
Traffic Light Detection Module - COLOR-BASED (No YOLO Model Required)
=====================================================================

Detects traffic light colors (Red, Green, Yellow) using HSV color analysis.
All parameters are imported from constants.py for easy adjustment.

Instead of using YOLO object detection:
- Analyzes the entire frame (or ROI region) for red/green pixel concentrations
- Compares pixel ratios against configurable thresholds
- No model inference needed - pure color analysis

Author: Auto Vehicle Control System
Date: 2026-05-18
"""

import cv2
import requests
import numpy as np
from collections import deque
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
    TRAFFIC_LIGHT_LOWER_YELLOW,
    TRAFFIC_LIGHT_UPPER_YELLOW,
    TRAFFIC_LIGHT_RED_RATIO_THRESHOLD,
    TRAFFIC_LIGHT_GREEN_RATIO_THRESHOLD,
    TRAFFIC_LIGHT_YELLOW_RATIO_THRESHOLD,
    TRAFFIC_LIGHT_ROI_ENABLED,
    TRAFFIC_LIGHT_ROI_START_RATIO,
    TRAFFIC_LIGHT_ROI_END_RATIO,
    TRAFFIC_LIGHT_STATE_HISTORY_SIZE,
    TRAFFIC_LIGHT_STATE_CONFIDENCE_THRESHOLD,
    COMMAND_STOP,
    COMMAND_CONTINUE,
    DEBUG_MODE,
    SHOW_VISUAL_OUTPUT,
)


class TrafficLightDetector:
    """
    Traffic light color detection using HSV color space analysis.
    
    No YOLO model needed - analyzes pixel color distribution directly.
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
        
        # ===== HSV Color Ranges =====
        # Red (split into two ranges due to HSV wraparound at 0/180)
        self.lower_red1 = TRAFFIC_LIGHT_LOWER_RED1
        self.upper_red1 = TRAFFIC_LIGHT_UPPER_RED1
        self.lower_red2 = TRAFFIC_LIGHT_LOWER_RED2
        self.upper_red2 = TRAFFIC_LIGHT_UPPER_RED2
        
        # Green
        self.lower_green = TRAFFIC_LIGHT_LOWER_GREEN
        self.upper_green = TRAFFIC_LIGHT_UPPER_GREEN
        
        # Yellow (optional, for monitoring)
        self.lower_yellow = TRAFFIC_LIGHT_LOWER_YELLOW
        self.upper_yellow = TRAFFIC_LIGHT_UPPER_YELLOW
        
        # ===== Pixel Ratio Thresholds =====
        # How much of the frame (or ROI) must be that color to trigger detection
        self.red_ratio_threshold = TRAFFIC_LIGHT_RED_RATIO_THRESHOLD
        self.green_ratio_threshold = TRAFFIC_LIGHT_GREEN_RATIO_THRESHOLD
        self.yellow_ratio_threshold = TRAFFIC_LIGHT_YELLOW_RATIO_THRESHOLD
        
        # ===== ROI Configuration =====
        # Optionally limit detection to a specific region (e.g., top 40% of frame)
        self.roi_enabled = TRAFFIC_LIGHT_ROI_ENABLED
        self.roi_start_ratio = TRAFFIC_LIGHT_ROI_START_RATIO
        self.roi_end_ratio = TRAFFIC_LIGHT_ROI_END_RATIO
        
        # ===== State Smoothing =====
        # Use history buffer to prevent flickering between states
        self.state_history = deque(maxlen=TRAFFIC_LIGHT_STATE_HISTORY_SIZE)
        self.confidence_threshold = TRAFFIC_LIGHT_STATE_CONFIDENCE_THRESHOLD
        
        # ===== State Tracking =====
        self.last_light_state = None
        self.red_light_detected = False
        self.traffic_light_state = None
        self.light_state_changed = False
        
        # ===== Statistics =====
        self.frame_count = 0
        self.traffic_lights_detected = 0
        
        # Print initialization message
        if DEBUG_MODE:
            print("\n[TL DETECTOR] COLOR-BASED Traffic Light Detector Initialized")
            print(f"[TL DETECTOR] Red threshold: {self.red_ratio_threshold:.1%}")
            print(f"[TL DETECTOR] Green threshold: {self.green_ratio_threshold:.1%}")
            print(f"[TL DETECTOR] Yellow threshold: {self.yellow_ratio_threshold:.1%}")
            if self.roi_enabled:
                print(f"[TL DETECTOR] ROI enabled: {self.roi_start_ratio:.0%} - {self.roi_end_ratio:.0%}")
            else:
                print(f"[TL DETECTOR] ROI disabled (full frame analysis)")
    
    def extract_roi(self, frame):
        """
        Extract region of interest from frame if enabled.
        
        Args:
            frame: Input frame
            
        Returns:
            np.ndarray: ROI frame (or full frame if ROI disabled)
        """
        if not self.roi_enabled:
            return frame
        
        h = frame.shape[0]
        start_y = int(h * self.roi_start_ratio)
        end_y = int(h * self.roi_end_ratio)
        
        return frame[start_y:end_y, :]
    
    def detect_light_color(self, frame):
        """
        Detect traffic light color in frame using HSV color analysis.
        
        Priority: RED > YELLOW > GREEN
        
        Args:
            frame: Input frame (or ROI)
            
        Returns:
            tuple: (color_name, red_ratio, green_ratio, yellow_ratio)
                   where color_name is "RED", "YELLOW", "GREEN", or None
        """
        if frame is None or frame.size == 0:
            return None, 0, 0, 0
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # ===== Detect Red (two ranges) =====
        mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        red_count = cv2.countNonZero(mask_red)
        
        # ===== Detect Yellow =====
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        yellow_count = cv2.countNonZero(mask_yellow)
        
        # ===== Detect Green =====
        mask_green = cv2.inRange(hsv, self.lower_green, self.upper_green)
        green_count = cv2.countNonZero(mask_green)
        
        # ===== Calculate Ratios =====
        total_pixels = frame.shape[0] * frame.shape[1]
        red_ratio = red_count / total_pixels if total_pixels > 0 else 0
        yellow_ratio = yellow_count / total_pixels if total_pixels > 0 else 0
        green_ratio = green_count / total_pixels if total_pixels > 0 else 0
        
        # Debug output
        if DEBUG_MODE and self.frame_count % 30 == 0:
            print(f"[TL COLOR] Frame {self.frame_count}: Red={red_ratio:.2%} | "
                  f"Yellow={yellow_ratio:.2%} | Green={green_ratio:.2%}")
        
        # ===== Priority Detection: RED > YELLOW > GREEN =====
        if red_ratio > self.red_ratio_threshold:
            return "RED", red_ratio, green_ratio, yellow_ratio
        elif yellow_ratio > self.yellow_ratio_threshold:
            return "YELLOW", red_ratio, green_ratio, yellow_ratio
        elif green_ratio > self.green_ratio_threshold:
            return "GREEN", red_ratio, green_ratio, yellow_ratio
        else:
            return None, red_ratio, green_ratio, yellow_ratio
    
    def smooth_state(self, current_color):
        """
        Apply state smoothing using history buffer to prevent flickering.
        
        Args:
            current_color: Current detected color
            
        Returns:
            str: Smoothed state (may differ from current_color if buffer not full)
        """
        self.state_history.append(current_color)
        
        if len(self.state_history) == 0:
            return None
        
        # Count occurrences of each state
        color_counts = {}
        for color in self.state_history:
            color_counts[color] = color_counts.get(color, 0) + 1
        
        # Find most common state
        most_common = max(color_counts.items(), key=lambda x: x[1])
        most_common_color, count = most_common
        
        # Check if confidence threshold is met
        confidence = count / len(self.state_history)
        if confidence >= self.confidence_threshold:
            return most_common_color
        else:
            return None
    
    def detect_state_change(self, current_color):
        """
        Detect if traffic light state changed (used for logging).
        
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
        Determine movement command based on traffic light color.
        
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
            LABEL_NAMES: List of class names (unused in color-based detection)
            exit_flag: Dict flag to signal thread exit {'flag': bool}
        """
        print("\n[TRAFFIC LIGHT] ========== COLOR-BASED TRAFFIC LIGHT DETECTOR ==========")
        print("[TRAFFIC LIGHT] Starting traffic light detection thread...")
        print(f"[TRAFFIC LIGHT] Detection method: COLOR ANALYSIS (No YOLO model)")
        print(f"[TRAFFIC LIGHT] Red threshold: {self.red_ratio_threshold:.1%}")
        print(f"[TRAFFIC LIGHT] Green threshold: {self.green_ratio_threshold:.1%}")
        print(f"[TRAFFIC LIGHT] Yellow threshold: {self.yellow_ratio_threshold:.1%}")
        print("[TRAFFIC LIGHT] ===================================================================\n")
        
        last_sent_command = None
        command_debounce_time = 0
        
        while not exit_flag['flag']:
            ret, frame = cap.read()
            if not ret:
                if DEBUG_MODE:
                    print("[TRAFFIC LIGHT] Failed to read frame")
                break
            
            self.frame_count += 1
            
            # Extract ROI if enabled
            analysis_frame = self.extract_roi(frame)
            
            # Detect light color
            light_color, red_ratio, green_ratio, yellow_ratio = self.detect_light_color(analysis_frame)
            
            # Apply state smoothing
            smoothed_color = self.smooth_state(light_color)
            
            # Update shared state
            with self.state_lock:
                if smoothed_color == "RED":
                    self.red_light_detected = True
                else:
                    self.red_light_detected = False
            
            # Send control command (with debouncing to avoid spam)
            if smoothed_color and (time.time() - command_debounce_time > 0.5 or 
                                    smoothed_color != self.last_light_state):
                command, status = self.get_command_by_light_color(smoothed_color)
                state_changed = self.detect_state_change(smoothed_color)
                
                self.send_command(command)
                command_debounce_time = time.time()
                
                if DEBUG_MODE or state_changed:
                    print(f"[TRAFFIC LIGHT] {status} | "
                          f"Red={red_ratio:.1%} Green={green_ratio:.1%} Yellow={yellow_ratio:.1%}")
                
                if state_changed:
                    print(f"[TRAFFIC LIGHT] ⚡ State changed: {self.last_light_state} → {smoothed_color}")
                
                self.last_light_state = smoothed_color
                last_sent_command = command
            
            # Visualization
            if SHOW_VISUAL_OUTPUT:
                try:
                    display_frame = frame.copy()
                    
                    # Convert to HSV for mask visualization
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    
                    # Create masks
                    mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
                    mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
                    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
                    
                    mask_green = cv2.inRange(hsv, self.lower_green, self.upper_green)
                    mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
                    
                    # Add text overlay
                    status_text = f"State: {self.last_light_state or 'NONE'}"
                    cv2.putText(display_frame, status_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    color_text = f"R:{red_ratio:.1%} G:{green_ratio:.1%} Y:{yellow_ratio:.1%}"
                    cv2.putText(display_frame, color_text, (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display original and masks side by side
                    display_red = cv2.cvtColor(mask_red, cv2.COLOR_GRAY2BGR)
                    display_green = cv2.cvtColor(mask_green, cv2.COLOR_GRAY2BGR)
                    display_yellow = cv2.cvtColor(mask_yellow, cv2.COLOR_GRAY2BGR)
                    
                    # Resize for display
                    h = display_frame.shape[0] // 2
                    display_frame = cv2.resize(display_frame, (320, h))
                    display_red = cv2.resize(display_red, (320, h))
                    display_green = cv2.resize(display_green, (320, h))
                    display_yellow = cv2.resize(display_yellow, (320, h))
                    
                    # Combine displays
                    top_row = cv2.hconcat([display_frame, display_red])
                    bottom_row = cv2.hconcat([display_green, display_yellow])
                    combined = cv2.vconcat([top_row, bottom_row])
                    
                    cv2.imshow('Traffic Light Detection (COLOR-BASED)', combined)
                    
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"[TRAFFIC LIGHT] Warning: Cannot display window - {str(e)}")
            
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
            'light_state_changed': self.light_state_changed,
            'frame_count': self.frame_count,
            'lights_detected': self.traffic_lights_detected
        }
