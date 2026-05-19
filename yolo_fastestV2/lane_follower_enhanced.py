"""
Enhanced Lane Detection and Following Module with ROI Focus and Visualization
================================================================================

Detects lane lines using HSV color detection on the BOTTOM 1/3 of the frame.
Determines vehicle movement commands based on lane curvature.
Displays complete frame with annotations showing:
- Lane positions and curvature
- Detected ROI area on full frame
- Curve radius calculation

All parameters are imported from constants.py.

Author: Auto Vehicle Control System
Date: 2026-05-18
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
    COMMAND_FORWARD,
    COMMAND_LEFT,
    COMMAND_RIGHT,
    DEBUG_MODE,
)


class LaneFollowerEnhanced:
    """Enhanced lane detection and following module with visualization"""
    
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
        
        # ROI settings - FOCUS ON BOTTOM 1/3 OF FRAME
        self.roi_start_ratio = 2/3  # Start at 2/3 from top (bottom 1/3)
        self.roi_end_ratio = 1.0    # End at bottom
        
        # State tracking
        self.last_lane_command = None
        self.red_light_detected = False
        self.zebra_crossing_detected = False
        
        # Lane tracking for curvature calculation
        self.prev_left_center = None
        self.prev_right_center = None
        self.prev_curvature = 0
        
        # Statistics
        self.frame_count = 0
    
    def detect_lane_roi(self, frame):
        """
        Detect lane lines in the BOTTOM 1/3 of the frame using HSV color thresholding
        
        Args:
            frame: Input frame from video capture
            
        Returns:
            tuple: (mask, frame_roi, roi_bounds) - binary mask, ROI frame, (start_y, end_y)
        """
        h, w = frame.shape[:2]
        
        # Extract bottom 1/3 of frame for processing
        roi_start = int(h * self.roi_start_ratio)
        roi_end = int(h * self.roi_end_ratio)
        frame_roi = frame[roi_start:roi_end, :]
        
        # Detect white lane lines using HSV
        hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # Morphological operations to clean up mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        
        return mask, frame_roi, (roi_start, roi_end)
    
    def calculate_curve_radius(self, left_center, right_center, roi_height):
        """
        Calculate the approximate radius of curvature for the lane
        
        Args:
            left_center: X position of left lane center
            right_center: X position of right lane center
            roi_height: Height of the ROI region
            
        Returns:
            float: Approximate curve radius in pixels (higher = gentler curve)
        """
        lane_width = right_center - left_center
        
        if lane_width <= 0:
            return float('inf')
        
        # Simplified curve radius calculation
        # Assume parabolic lane: R = (lane_width^2 + roi_height^2) / (2 * lane_width)
        curve_radius = (lane_width ** 2 + roi_height ** 2) / (2 * abs(lane_width))
        return curve_radius
    
    def find_lane_curvature(self, mask, frame):
        """
        Calculate lane curvature and offset from center
        
        Args:
            mask: Binary mask of lane (from bottom 1/3)
            frame: Original full frame (for dimension reference)
            
        Returns:
            tuple: (curvature, offset, left_center, right_center, curve_radius)
        """
        h, w = frame.shape[:2]
        mask_h, mask_w = mask.shape
        
        # Find non-zero pixels in mask
        cols = cv2.findNonZero(mask)
        
        if cols is None or len(cols) < 10:
            return 0, 0, None, None, float('inf')
        
        # Extract x coordinates of detected pixels
        x_coords = cols[:, 0, 0]
        mid_w = mask_w // 2
        
        # Split into left and right lane markers
        left_points = x_coords[x_coords < mid_w]
        right_points = x_coords[x_coords >= mid_w]
        
        # Calculate lane centers
        left_center = np.mean(left_points) if len(left_points) > 0 else mid_w * 0.25
        right_center = np.mean(right_points) if len(right_points) > 0 else mid_w * 1.75
        
        # Calculate offset from image center
        lane_center = (left_center + right_center) / 2
        image_center = mask_w / 2
        offset = (lane_center - image_center) / image_center
        
        # Calculate curvature based on lane width
        lane_width = right_center - left_center
        curvature = offset * (mask_w / (lane_width + 1e-5))
        
        # Calculate approximate curve radius
        curve_radius = self.calculate_curve_radius(left_center, right_center, mask_h)
        
        # Store for smoothing
        self.prev_left_center = left_center
        self.prev_right_center = right_center
        self.prev_curvature = curvature
        
        return curvature, offset, left_center, right_center, curve_radius
    
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
    
    def should_execute(self, red_light_detected, zebra_crossing_detected=False):
        """
        Check if lane following should execute based on priority system.
        
        Priority: red_light > zebra_crossing > lane_following
        
        Args:
            red_light_detected: Whether red light was detected
            zebra_crossing_detected: Whether zebra crossing was detected
            
        Returns:
            bool: True if lane following should be executed
        """
        # When red light is detected, STOP (don't execute lane following)
        if red_light_detected:
            return False
        
        # When zebra crossing is detected, CONTINUE lane following (safe passage)
        # Only execute lane following if no red light detected
        return True
    
    def annotate_frame(self, frame, roi_bounds, mask, left_center, right_center, 
                      curve_radius, curvature, offset, command, status):
        """
        Annotate the full frame with detection results
        
        Args:
            frame: Original full frame
            roi_bounds: (roi_start_y, roi_end_y) tuple
            mask: Binary mask of detected lanes
            left_center: X position of left lane
            right_center: X position of right lane
            curve_radius: Calculated curve radius
            curvature: Lane curvature value
            offset: Lane offset
            command: Movement command
            status: Status string
            
        Returns:
            np.ndarray: Annotated frame
        """
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        roi_start, roi_end = roi_bounds
        
        # Draw ROI rectangle on full frame (highlight the bottom 1/3)
        cv2.rectangle(display_frame, (0, roi_start), (w, roi_end), (255, 255, 0), 2)
        cv2.putText(display_frame, "DETECTION ROI (Bottom 1/3)", (10, roi_start - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw lane positions if detected
        if left_center is not None and right_center is not None:
            # Convert mask coordinates to full frame coordinates
            roi_height = roi_end - roi_start
            
            # Draw vertical lines for lane positions on full frame
            left_x = int(left_center)
            right_x = int(right_center)
            
            # Draw lane lines on the ROI area of full frame
            cv2.line(display_frame, (left_x, roi_start), (left_x, roi_end), (0, 255, 0), 2)
            cv2.line(display_frame, (right_x, roi_start), (right_x, roi_end), (0, 255, 0), 2)
            
            # Draw lane center
            lane_center = (left_x + right_x) // 2
            cv2.line(display_frame, (lane_center, roi_start), (lane_center, roi_end), (255, 0, 0), 2)
            
            # Add lane information
            cv2.putText(display_frame, f"Left Lane: {left_x}px", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Right Lane: {right_x}px", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add curve information
            curve_text = f"Curve Radius: {curve_radius:.0f}px"
            if np.isinf(curve_radius):
                curve_text = "Curve Radius: Straight"
            cv2.putText(display_frame, curve_text, (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add status information
        cv2.putText(display_frame, f"Curvature: {curvature:.2f}", (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(display_frame, f"Offset: {offset:.2f}", (10, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(display_frame, f"Status: {status}", (10, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Command: {command}", (10, 230),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if self.red_light_detected:
            cv2.putText(display_frame, "[RED LIGHT - STOP]", (10, 260),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif self.zebra_crossing_detected:
            cv2.putText(display_frame, "[ZEBRA CROSSING - CONTINUE]", (10, 260),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        return display_frame
    
    def get_state(self):
        """
        Get current lane following state
        
        Returns:
            dict: Current state information
        """
        return {
            'last_lane_command': self.last_lane_command,
            'red_light_detected': self.red_light_detected,
            'zebra_crossing_detected': self.zebra_crossing_detected,
            'frame_count': self.frame_count
        }
    
    def print_thresholds(self):
        """Print current lane detection thresholds"""
        print("\n[LANE] Lane Detection Thresholds:")
        print(f"  Sharp Left:  < {self.sharp_left_threshold}")
        print(f"  Gentle Left: < {self.gentle_left_threshold}")
        print(f"  Gentle Right: > {self.gentle_right_threshold}")
        print(f"  Sharp Right: > {self.sharp_right_threshold}")
        print(f"  ROI: Bottom 1/3 of frame (from {self.roi_start_ratio*100:.0f}% to {self.roi_end_ratio*100:.0f}%)")
