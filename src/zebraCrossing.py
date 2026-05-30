import cv2
import numpy as np
import requests
from threading import Lock
from collections import deque
from Car_state import Car_State


class ZebraCrossing:
    
    def __init__(self, frame, car_state):
        self.frame = frame
        # Zebra crossing detection parameters - 使用与LaneKeepingAlgorithm相同的HSV数值
        self.lower_white = np.array([0, 0, 180])
        self.upper_white = np.array([180, 40, 255])
        
        # Detection thresholds
        # self.horizontal_ratio_threshold = ZEBRA_CROSSING_HORIZONTAL_RATIO_THRESHOLD
        # self.pattern_threshold = ZEBRA_CROSSING_PATTERN_THRESHOLD
        # self.confidence_threshold = ZEBRA_CROSSING_CONFIDENCE_THRESHOLD
        
        # ROI设置 - 画面最下面一半的中间一半部分（去除左右各1/4）
        self.roi_start_ratio = 0.5   # 从画面50%高度开始（下半部分）
        self.roi_end_ratio = 1.0     # 到画面底部
        self.roi_left_ratio = 0.25   # 左边1/4不要
        self.roi_right_ratio = 0.75  # 右边1/4不要（保留中间50%）
        
        # 面积阈值 - 白色像素数量大于此值才认为是斑马线
        self.area_threshold = 1000
        
        # 状态管理
        self.car_state = car_state  # 默认是车道保持状态
        
        # State tracking
        self.zebra_crossing_detected = False
        self.crossing_state = None
        
        
    def CROSSING_detect_zebra_crossing(self):
        """
        Detect zebra crossing patterns in frame using HSV color thresholding
        Only analyzes the bottom half of the frame, middle 50% (excluding left/right 1/4)
        
        Args:
            frame: Input frame from video capture
            
        Returns:
            tuple: (mask_full, mask_roi, roi_y_start) - full frame mask, ROI mask, start y
        """
        h, w = self.frame.shape[:2]
        
        # Apply ROI - 最下面一半的中间一半部分（去除左右各1/4）
        roi_start = int(h * self.roi_start_ratio)
        roi_end = int(h * self.roi_end_ratio)
        roi_left = int(w * self.roi_left_ratio)
        roi_right = int(w * self.roi_right_ratio)
        
        frame_roi = self.frame[roi_start:roi_end, roi_left:roi_right]
        
        # Convert to HSV and detect white stripes
        hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
        mask_roi = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # Morphological operations to enhance stripe pattern
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 计算白se区域面积
        white_pixel_count = cv2.countNonZero(mask_roi)
        
        # 创建完整高度的掩码用于可视化
        mask_full = np.zeros((h, w), dtype=np.uint8)
        mask_full[roi_start:roi_end, roi_left:roi_right] = mask_roi
        
        return white_pixel_count
    
    def CROSSING_determine_state(self, white_pixel_count):
        if white_pixel_count >= self.area_threshold:
            self.zebra_crossing_detected = True
            self.car_state = Car_State.WAIT
        else:
            self.zebra_crossing_detected = False

    def CROSSING_main(self):
        white_pixel_count = self.CROSSING_detect_zebra_crossing()
        self.CROSSING_determine_state(white_pixel_count)
"""       
        # Count horizontal white pixels per row
        horizontal_counts = np.sum(mask, axis=1)
        total_pixels = w
        
        # Find rows with significant white content (potential stripes)
        stripe_rows = horizontal_counts > (total_pixels * self.horizontal_ratio_threshold)
        stripe_count = np.sum(stripe_rows)
        
        # Detect alternating stripe pattern (zebra crossing characteristic)
        if stripe_count > 0:
            # Calculate pattern regularity (alternating dark and light regions)
            stripe_indices = np.where(stripe_rows)[0]
            
            if len(stripe_indices) > self.pattern_threshold:
                # Check for regular spacing between stripes
                stripe_gaps = np.diff(stripe_indices)
                regular_spacing = np.std(stripe_gaps) < np.mean(stripe_gaps) * 0.5
                
                if regular_spacing:
                    # Calculate confidence based on number of stripes and regularity
                    confidence = min(1.0, stripe_count / (self.pattern_threshold * 2))
                    return True, stripe_count, confidence
        
        return False, stripe_count, 0.0
"""
    