"""
Lane Detection and Following Module with Zebra Crossing Awareness
===================================================================

Detects lane lines using HSV color detection and determines vehicle movement commands
based on lane curvature. Integrates with zebra crossing detection to maintain
lane following through crossings.

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
    LANE_ROI_START_RATIO,
    COMMAND_FORWARD,
    COMMAND_LEFT,
    COMMAND_RIGHT,
    DEBUG_MODE,
)


class LaneFollower:
    """Lane detection and following module with zebra crossing awareness"""
    
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
        
        # ROI settings - 只在最下面的1/3部分检测
        self.roi_start_ratio = LANE_ROI_START_RATIO  # 从2/3处开始
        
        # State tracking
        self.last_lane_command = None
        self.red_light_detected = False
        self.zebra_crossing_detected = False  # NEW: Zebra crossing awareness
        
        # Lane tracking info for visualization
        self.left_lane_x = []
        self.right_lane_x = []
        self.lane_curvature = 0
        self.lane_offset = 0
        self.lane_radius = 0  # 曲率半径
        
        # Statistics
        self.frame_count = 0
    
    def detect_lane(self, frame):
        """
        Detect lane lines in frame using HSV color thresholding
        Only analyzes the bottom 1/3 of the frame
        
        Args:
            frame: Input frame from video capture
            
        Returns:
            tuple: (mask, roi_mask, roi_y_start) - binary mask, ROI-only mask, ROI start y
        """
        h, w = frame.shape[:2]
        
        # Apply ROI - 只在最下面的1/3部分
        roi_start_y = int(h * self.roi_start_ratio)
        frame_roi = frame[roi_start_y:, :]
        
        hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
        mask_roi = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # Morphological operations to clean up mask
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, self.kernel)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, self.kernel)
        
        # 创建完整高度的掩码用于可视化
        mask_full = np.zeros((h, w), dtype=np.uint8)
        mask_full[roi_start_y:, :] = mask_roi
        
        return mask_full, mask_roi, roi_start_y
    
    def find_lane_curvature(self, mask_roi, frame, roi_y_start):
        """
        Calculate lane curvature, offset and radius
        
        Args:
            mask_roi: Binary mask of lane (ROI region only)
            frame: Original frame (for dimension reference)
            roi_y_start: Starting Y coordinate of ROI
            
        Returns:
            tuple: (curvature, offset, radius, left_x, right_x) - normalized values
        """
        h_roi, w = mask_roi.shape
        h_full = frame.shape[0]
        
        cols = cv2.findNonZero(mask_roi)
        
        if cols is None or len(cols) < 10:
            return 0, 0, 0, [], []
        
        # Extract x coordinates of detected pixels
        x_coords = cols[:, 0, 0]
        y_coords = cols[:, 0, 1]
        mid_w = w // 2
        
        # Split into left and right lane markers
        left_points = x_coords[x_coords < mid_w]
        right_points = x_coords[x_coords >= mid_w]
        
        left_y = y_coords[x_coords < mid_w]
        right_y = y_coords[x_coords >= mid_w]
        
        # Calculate lane centers
        left_center = np.mean(left_points) if len(left_points) > 0 else mid_w * 0.25
        right_center = np.mean(right_points) if len(right_points) > 0 else mid_w * 1.75
        
        # 保存检测到的车道点用于可视化
        self.left_lane_x = sorted(set(left_points.astype(int).tolist())) if len(left_points) > 0 else []
        self.right_lane_x = sorted(set(right_points.astype(int).tolist())) if len(right_points) > 0 else []
        
        # Calculate offset from image center
        lane_center = (left_center + right_center) / 2
        image_center = w / 2
        offset = (lane_center - image_center) / image_center
        
        # Calculate curvature based on lane width
        lane_width = right_center - left_center
        curvature = offset * (w / (lane_width + 1e-5))
        
        # 计算曲率半径 (Radius of Curvature)
        # 使用简单的曲率公式: R = (1 + (dy/dx)^2)^1.5 / |d2y/dx2|
        if len(left_points) > 2 and len(right_points) > 2:
            try:
                # 拟合左右车道线的二次多项式
                left_fit = np.polyfit(left_y, left_points, 2) if len(left_y) > 2 else None
                right_fit = np.polyfit(right_y, right_points, 2) if len(right_y) > 2 else None
                
                if left_fit is not None:
                    # 计算二阶导数 (d2y/dx2 = 2*a, 其中ax^2+bx+c)
                    a = left_fit[0]
                    # 半径 = ((1 + b^2)^1.5) / |2*a| * pixels_per_meter
                    # 简化: R = 1 / (2*|a|)
                    if abs(a) > 1e-5:
                        radius = 1.0 / (2 * abs(a))
                    else:
                        radius = float('inf')
                else:
                    radius = float('inf')
            except:
                radius = float('inf')
        else:
            radius = float('inf')
        
        self.lane_curvature = curvature
        self.lane_offset = offset
        self.lane_radius = radius if radius != float('inf') else 0
        
        return curvature, offset, radius, left_center, right_center
    
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
        
        NEW priority system (Person detection disabled):
        Priority: red_light > zebra_crossing > lane_following
        
        Args:
            red_light_detected: Whether red light was detected
            zebra_crossing_detected: Whether zebra crossing was detected
            
        Returns:
            bool: True if lane following should be executed
        """
        # ⚠️  IMPORTANT: When red light is detected, STOP (don't execute lane following)
        if red_light_detected:
            return False
        
        # ✅ NEW: When zebra crossing is detected, CONTINUE lane following (safe passage)
        # This ensures vehicle maintains lane discipline through the crossing
        # (zebra_crossing_detected doesn't block lane following)
        
        # Only execute lane following if no red light detected
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
            'zebra_crossing_detected': self.zebra_crossing_detected,
            'frame_count': self.frame_count,
            'curvature': self.lane_curvature,
            'offset': self.lane_offset,
            'radius': self.lane_radius
        }
    
    def print_thresholds(self):
        """Print current lane detection thresholds"""
        print("\n[LANE] Lane Detection Thresholds:")
        print(f"  Sharp Left:  < {self.sharp_left_threshold}")
        print(f"  Gentle Left: < {self.gentle_left_threshold}")
        print(f"  Gentle Right: > {self.gentle_right_threshold}")
        print(f"  Sharp Right: > {self.sharp_right_threshold}")
        print(f"  ROI Start: {self.roi_start_ratio*100:.0f}% of frame height (bottom 1/3)")
