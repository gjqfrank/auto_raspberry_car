"""
Lane Detection and Following Module
====================================

Detects lane lines using HSV color detection and determines vehicle movement commands
based on lane curvature. All parameters are imported from constants.py.

Author: Auto Vehicle Control System
Date: 2026-05-13
"""

import cv2
import numpy as np
import requests
from threading import Lock
from constants import (
    CONTROL_URL,
    LANE_LOWER_WHITE,
    LANE_UPPER_WHITE,
    LANE_SHARP_LEFT_THRESHOLD,
    LANE_GENTLE_LEFT_THRESHOLD,
    LANE_GENTLE_RIGHT_THRESHOLD,
    LANE_SHARP_RIGHT_THRESHOLD,
    LANE_KERNEL_SIZE,
    LANE_ROI_START_RATIO,
    COMMAND_FORWARD,
    COMMAND_LEFT,
    COMMAND_RIGHT,
    DEBUG_MODE,
)


class LaneFollower:
    """Lane detection and following module"""
    
    def __init__(self, control_url, state_lock):
        self.control_url = control_url
        self.state_lock = state_lock
        
        # Lane detection parameters
        self.lower_white = LANE_LOWER_WHITE
        self.upper_white = LANE_UPPER_WHITE
        
        # Lane thresholds for direction determination
        self.sharp_left_threshold = LANE_SHARP_LEFT_THRESHOLD
        self.gentle_left_threshold = LANE_GENTLE_LEFT_THRESHOLD
        self.gentle_right_threshold = LANE_GENTLE_RIGHT_THRESHOLD
        self.sharp_right_threshold = LANE_SHARP_RIGHT_THRESHOLD
        
        # Morphological operations kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, LANE_KERNEL_SIZE)
        
        # ROI settings
        self.roi_start_ratio = LANE_ROI_START_RATIO
        
        # State tracking
        self.last_lane_command = None
        self.red_light_detected = False
        self.person_detected = False
        self.person_in_danger_zone = False
        
        # Statistics
        self.frame_count = 0
    
    def detect_lane(self, frame):
        """
        Detect lane lines in frame using HSV color thresholding
        
        Args:
            frame: Input frame from video capture
            
        Returns:
            np.ndarray: Binary mask of detected lane lines
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # Morphological operations to clean up mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        
        return mask
    
    def find_lane_curvature(self, mask, frame):
        """
        Calculate lane curvature and offset from center
        
        Args:
            mask: Binary mask of lane
            frame: Original frame (for dimension reference)
            
        Returns:
            tuple: (curvature, offset) - normalized values for control
        """
        h, w = mask.shape
        
        # Focus on bottom half of image (where car sees road ahead)
        mask_roi = mask[int(h * self.roi_start_ratio):, :]
        cols = cv2.findNonZero(mask_roi)
        
        if cols is None or len(cols) < 10:
            return 0, 0
        
        # Extract x coordinates of detected pixels
        x_coords = cols[:, 0, 0]
        mid_w = w // 2
        
        # Split into left and right lane markers
        left_points = x_coords[x_coords < mid_w]
        right_points = x_coords[x_coords >= mid_w]
        
        # Calculate lane centers
        left_center = np.mean(left_points) if len(left_points) > 0 else mid_w * 0.25
        right_center = np.mean(right_points) if len(right_points) > 0 else mid_w * 1.75
        
        # Calculate offset from image center
        lane_center = (left_center + right_center) / 2
        image_center = w / 2
        offset = (lane_center - image_center) / image_center
        
        # Calculate curvature based on lane width
        lane_width = right_center - left_center
        curvature = offset * (w / (lane_width + 1e-5))
        
        return curvature, offset
    
    def get_command_by_curvature(self, curvature, offset):
        """
        Determine movement command based on lane curvature
        
        Args:
            curvature: Lane curvature value
            offset: Lane offset from center
            
        Returns:
            tuple: (command, status, curvature, offset)
        """
        if curvature < self.sharp_left_threshold:
            command = COMMAND_LEFT
            status = "Sharp Left Turn"
        elif curvature < self.gentle_left_threshold:
            command = COMMAND_FORWARD
            status = "Gentle Left Turn"
        elif curvature > self.sharp_right_threshold:
            command = COMMAND_RIGHT
            status = "Sharp Right Turn"
        elif curvature > self.gentle_right_threshold:
            command = COMMAND_FORWARD
            status = "Gentle Right Turn"
        else:
            command = COMMAND_FORWARD
            status = "Going Straight"
        
        return command, status, curvature, offset
    
    def send_command(self, command):
        """
        Send movement command to vehicle
        
        Args:
            command: Command string (FORWARD, LEFT, RIGHT, etc.)
            
        Returns:
            bool: True if command sent successfully
        """
        try:
            response = requests.post(self.control_url, json={'command': command}, timeout=2)
            return response.status_code == 200
        except Exception as e:
            print(f"[LANE] ❌ Error sending command: {str(e)}")
            return False
    
    def should_execute(self, red_light_detected, person_detected, person_in_danger_zone=False):
        """
        Check if lane following should execute based on priority system
        
        Priority: red_light > person_danger > person_detected > lane_following
        
        Args:
            red_light_detected: Whether red light was detected
            person_detected: Whether person was detected
            person_in_danger_zone: Whether person is too close
            
        Returns:
            bool: True if lane following should be executed
        """
        # Emergency stop has highest priority
        if person_in_danger_zone:
            return False
        
        # Traffic light has priority
        if red_light_detected:
            return False
        
        # Person detection has priority
        if person_detected:
            return False
        
        # Only execute lane following if all higher priority conditions are false
        return True
    
    def get_state(self):
        """
        Get current lane following state
        
        Returns:
            dict: Current state information
        """
        return {
            'last_lane_command': self.last_lane_command,
            'red_light_detected': self.red_light_detected,
            'person_detected': self.person_detected,
            'person_in_danger_zone': self.person_in_danger_zone,
            'frame_count': self.frame_count
        }
    
    def print_thresholds(self):
        """Print current lane detection thresholds"""
        print("\n[LANE] Lane Detection Thresholds:")
        print(f"  Sharp Left:  < {self.sharp_left_threshold}")
        print(f"  Gentle Left: < {self.gentle_left_threshold}")
        print(f"  Gentle Right: > {self.gentle_right_threshold}")
        print(f"  Sharp Right: > {self.sharp_right_threshold}")
        print(f"  ROI Start: {self.roi_start_ratio*100:.0f}% of frame height")
