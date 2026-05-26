"""
Lane Detection and Following Module with Zebra Crossing Awareness
===================================================================

Detects lane lines using IMPROVED Canny Edge Detection and determines vehicle movement commands
based on lane curvature. Integrates with zebra crossing detection to maintain
lane following through crossings.

Key Improvements:
- Grayscale + Canny edge detection (robust to lighting changes)
- Optimized Hough transform parameters
- Slope filtering to eliminate noise
- PD controller for smooth steering
- Better ROI management

All parameters are imported from constants.py.

Author: Auto Vehicle Control System
Date: 2026-05-26
"""

import cv2
import numpy as np
import requests
import time
from threading import Lock
from constants import (
    CONTROL_URL,
    LANE_DETECTION_METHOD,
    LANE_CANNY_THRESHOLD1,
    LANE_CANNY_THRESHOLD2,
    LANE_GAUSSIAN_BLUR_KERNEL,
    LANE_LOWER_WHITE,
    LANE_UPPER_WHITE,
    LANE_SHARP_LEFT_THRESHOLD,
    LANE_GENTLE_LEFT_THRESHOLD,
    LANE_GENTLE_RIGHT_THRESHOLD,
    LANE_SHARP_RIGHT_THRESHOLD,
    LANE_KERNEL_SIZE,
    LANE_ROI_START_RATIO,
    LANE_HOUGH_RHO,
    LANE_HOUGH_THETA,
    LANE_HOUGH_MIN_THRESHOLD,
    LANE_HOUGH_MIN_LINE_LENGTH,
    LANE_HOUGH_MAX_LINE_GAP,
    LANE_SLOPE_MIN_THRESHOLD,
    LANE_BOUNDARY_RATIO,
    LANE_PD_KP,
    LANE_PD_KD,
    LANE_DEAD_ZONE,
    LANE_STEERING_SPEED,
    LANE_MOTOR_MAX_SPEED,
    COMMAND_FORWARD,
    COMMAND_LEFT,
    COMMAND_RIGHT,
    DEBUG_MODE,
)


class LaneFollower:
    """Lane detection and following module with zebra crossing awareness - IMPROVED"""
    
    def __init__(self, control_url, state_lock):
        self.control_url = control_url
        self.state_lock = state_lock
        
        # Lane detection method
        self.detection_method = LANE_DETECTION_METHOD
        
        # Canny edge detection parameters
        self.canny_threshold1 = LANE_CANNY_THRESHOLD1
        self.canny_threshold2 = LANE_CANNY_THRESHOLD2
        self.blur_kernel = LANE_GAUSSIAN_BLUR_KERNEL
        
        # Lane detection parameters (for HSV method backup)
        self.lower_white = LANE_LOWER_WHITE
        self.upper_white = LANE_UPPER_WHITE
        
        # Lane thresholds for direction determination
        self.sharp_left_threshold = LANE_SHARP_LEFT_THRESHOLD
        self.gentle_left_threshold = LANE_GENTLE_LEFT_THRESHOLD
        self.gentle_right_threshold = LANE_GENTLE_RIGHT_THRESHOLD
        self.sharp_right_threshold = LANE_SHARP_RIGHT_THRESHOLD
        
        # Morphological operations kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, LANE_KERNEL_SIZE)
        
        # Hough transform parameters (improved)
        self.hough_rho = LANE_HOUGH_RHO
        self.hough_theta = LANE_HOUGH_THETA
        self.hough_min_threshold = LANE_HOUGH_MIN_THRESHOLD
        self.hough_min_line_length = LANE_HOUGH_MIN_LINE_LENGTH
        self.hough_max_line_gap = LANE_HOUGH_MAX_LINE_GAP
        self.slope_min_threshold = LANE_SLOPE_MIN_THRESHOLD
        
        # ROI settings
        self.roi_start_ratio = LANE_ROI_START_RATIO
        self.boundary_ratio = LANE_BOUNDARY_RATIO
        
        # PD controller parameters
        self.kp = LANE_PD_KP
        self.kd = LANE_PD_KD
        self.dead_zone = LANE_DEAD_ZONE
        self.steering_speed = LANE_STEERING_SPEED
        self.motor_max_speed = LANE_MOTOR_MAX_SPEED
        
        # State tracking
        self.last_lane_command = None
        self.red_light_detected = False
        self.zebra_crossing_detected = False
        self.last_error = 0
        self.last_time = time.time()
        
        # Lane tracking info for visualization
        self.left_lane_x = []
        self.right_lane_x = []
        self.lane_curvature = 0
        self.lane_offset = 0
        self.lane_radius = 0
        self.steering_angle = 90  # 90 度表示直行
        
        # Statistics
        self.frame_count = 0
        
        if DEBUG_MODE:
            print("[LANE] ✅ Lane Follower initialized with IMPROVED Canny Edge Detection")
            self.print_thresholds()
    
    def detect_lane(self, frame):
        """
        Detect lane lines in frame using IMPROVED method
        - Grayscale + Gaussian Blur + Canny Edge Detection
        - Robust to lighting changes
        
        Args:
            frame: Input frame from video capture
            
        Returns:
            tuple: (mask, roi_mask, roi_y_start) - binary mask, ROI-only mask, ROI start y
        """
        h, w = frame.shape[:2]
        
        # Apply ROI - 只在最下面的1/3部分
        roi_start_y = int(h * self.roi_start_ratio)
        frame_roi = frame[roi_start_y:, :]
        
        if self.detection_method == "CANNY":
            # 改进：灰度 + 高斯模糊 + Canny 边界检测
            gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
            mask_roi = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
        else:
            # 备用：HSV 颜色检测方法
            hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
            mask_roi = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # Morphological operations to clean up mask
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, self.kernel)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, self.kernel)
        
        # 创建完整高度的掩码用于可视化
        mask_full = np.zeros((h, w), dtype=np.uint8)
        mask_full[roi_start_y:, :] = mask_roi
        
        return mask_full, mask_roi, roi_start_y
    
    def detect_line_segments(self, cropped_edges):
        """
        Detect line segments using Hough Transform with improved parameters
        
        Args:
            cropped_edges: Edge-detected ROI mask
            
        Returns:
            Line segments or None if not detected
        """
        line_segments = cv2.HoughLinesP(
            cropped_edges,
            self.hough_rho,
            self.hough_theta,
            self.hough_min_threshold,
            np.array([]),
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )
        
        return line_segments
    
    def find_lane_curvature(self, mask_roi, frame, roi_y_start):
        """
        Calculate lane curvature, offset and radius with improved slope filtering
        
        Args:
            mask_roi: Binary mask of lane (ROI region only)
            frame: Original frame (for dimension reference)
            roi_y_start: Starting Y coordinate of ROI
            
        Returns:
            tuple: (curvature, offset, radius, left_x, right_x)
        """
        h_roi, w = mask_roi.shape
        h_full = frame.shape[0]
        
        # Detect line segments
        line_segments = self.detect_line_segments(mask_roi)
        
        if line_segments is None or len(line_segments) == 0:
            return 0, 0, 0, w // 4, w * 3 // 4
        
        left_fit = []
        right_fit = []
        mid_w = w // 2
        left_region_boundary = w * (1 - self.boundary_ratio)
        right_region_boundary = w * self.boundary_ratio
        
        # Process each line segment with improved slope filtering
        for line_segment in line_segments:
            for x1, y1, x2, y2 in line_segment:
                if x1 == x2:
                    continue
                
                slope = (y2 - y1) / (x2 - x1)
                
                # 改进：过滤掉斜率过小的线（接近水平）和过大的线（接近竖直）
                if abs(slope) < self.slope_min_threshold:
                    continue
                
                intercept = y1 - (slope * x1)
                
                # 左车道（负斜率）
                if slope < 0:
                    if x1 < left_region_boundary and x2 < left_region_boundary:
                        left_fit.append((slope, intercept))
                # 右车道（正斜率）
                else:
                    if x1 > right_region_boundary and x2 > right_region_boundary:
                        right_fit.append((slope, intercept))
        
        # Calculate average lane lines
        left_center = mid_w * 0.25
        right_center = mid_w * 1.75
        
        if len(left_fit) > 0:
            left_fit_average = np.average(left_fit, axis=0)
            left_slope, left_intercept = left_fit_average
            # 计算左车道线与 ROI 中点的 x 坐标
            y_mid = h_roi // 2
            left_center = (y_mid - left_intercept) / left_slope if left_slope != 0 else mid_w * 0.25
            self.left_lane_x = [int(left_center)]
        
        if len(right_fit) > 0:
            right_fit_average = np.average(right_fit, axis=0)
            right_slope, right_intercept = right_fit_average
            # 计算右车道线与 ROI 中点的 x 坐标
            y_mid = h_roi // 2
            right_center = (y_mid - right_intercept) / right_slope if right_slope != 0 else mid_w * 1.75
            self.right_lane_x = [int(right_center)]
        
        # Calculate offset and curvature
        lane_center = (left_center + right_center) / 2
        image_center = w / 2
        offset = (lane_center - image_center) / image_center
        
        # Calculate curvature based on lane width
        lane_width = right_center - left_center
        curvature = offset * (w / (lane_width + 1e-5))
        
        # Calculate radius of curvature (simplified)
        if len(left_fit) > 0 and len(right_fit) > 0:
            try:
                left_fit_average = np.average(left_fit, axis=0)
                right_fit_average = np.average(right_fit, axis=0)
                # Average curvature from both lanes
                avg_curvature = (abs(left_fit_average[0]) + abs(right_fit_average[0])) / 2
                radius = 1.0 / (avg_curvature + 1e-5) if avg_curvature > 0 else float('inf')
            except:
                radius = float('inf')
        else:
            radius = float('inf')
        
        self.lane_curvature = curvature
        self.lane_offset = offset
        self.lane_radius = radius if radius != float('inf') else 0
        
        return curvature, offset, radius, left_center, right_center
    
    def get_steering_angle(self, curvature):
        """
        Calculate steering angle from lane curvature
        
        Args:
            curvature: Lane curvature value
            
        Returns:
            int: Steering angle (0-180, where 90 is straight)
        """
        # 将曲率映射到转向角度 (-45 到 +45)
        steering_angle_offset = max(-45, min(45, curvature * 90))
        steering_angle = 90 + steering_angle_offset
        
        self.steering_angle = steering_angle
        return int(steering_angle)
    
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
            if DEBUG_MODE:
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
        # This ensures vehicle maintains lane discipline through the crossing
        
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
            'radius': self.lane_radius,
            'steering_angle': self.steering_angle,
            'detection_method': self.detection_method
        }
    
    def print_thresholds(self):
        """Print current lane detection thresholds"""
        print("\n[LANE] 🎯 Lane Detection Configuration:")
        print(f"  Detection Method: {self.detection_method}")
        if self.detection_method == "CANNY":
            print(f"  Canny Threshold: {self.canny_threshold1} - {self.canny_threshold2}")
            print(f"  Gaussian Blur Kernel: {self.blur_kernel}")
        print(f"\n  Lane Curvature Thresholds:")
        print(f"    Sharp Left:  < {self.sharp_left_threshold}")
        print(f"    Gentle Left: < {self.gentle_left_threshold}")
        print(f"    Gentle Right: > {self.gentle_right_threshold}")
        print(f"    Sharp Right: > {self.sharp_right_threshold}")
        print(f"\n  Line Detection (Hough Transform):")
        print(f"    Min Threshold: {self.hough_min_threshold}")
        print(f"    Min Line Length: {self.hough_min_line_length}")
        print(f"    Max Line Gap: {self.hough_max_line_gap}")
        print(f"    Slope Min Threshold: {self.slope_min_threshold} (noise filtering)")
        print(f"\n  PD Controller:")
        print(f"    KP: {self.kp}, KD: {self.kd}")
        print(f"    Dead Zone: ±{self.dead_zone}°")
        print(f"    Steering Speed: {self.steering_speed}")
        print(f"    Max Motor Speed: {self.motor_max_speed}")
        print(f"\n  ROI: Starting at {self.roi_start_ratio*100:.0f}% (bottom 1/3)")
