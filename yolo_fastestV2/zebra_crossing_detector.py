"""
Zebra Crossing Detection Module
================================

Detects zebra crossing patterns using HSV color detection and stripe analysis.
When zebra crossing is detected, lane follower continues lane following for safe passage.

All parameters are imported from constants.py.

Author: Auto Vehicle Control System
Date: 2026-05-18
"""

import cv2
import numpy as np
import requests
from threading import Lock
from collections import deque
from constants import (
    CONTROL_URL,
    ZEBRA_LOWER_WHITE,
    ZEBRA_UPPER_WHITE,
    ZEBRA_CROSSING_HORIZONTAL_RATIO_THRESHOLD,
    ZEBRA_CROSSING_PATTERN_THRESHOLD,
    ZEBRA_CROSSING_CONFIDENCE_THRESHOLD,
    ZEBRA_CROSSING_ROI_START_RATIO,
    ZEBRA_CROSSING_ROI_END_RATIO,
    ZEBRA_CROSSING_STATE_HISTORY_SIZE,
    ZEBRA_CROSSING_STATE_CONFIDENCE_THRESHOLD,
    DEBUG_MODE,
    LOG_INTERVAL,
)


class ZebraCrossingDetector:
    """Zebra crossing detection module with lane-following integration"""
    
    def __init__(self, control_url, state_lock):
        self.control_url = control_url
        self.state_lock = state_lock
        
        # Zebra crossing detection parameters
        self.lower_white = ZEBRA_LOWER_WHITE
        self.upper_white = ZEBRA_UPPER_WHITE
        
        # Detection thresholds
        self.horizontal_ratio_threshold = ZEBRA_CROSSING_HORIZONTAL_RATIO_THRESHOLD
        self.pattern_threshold = ZEBRA_CROSSING_PATTERN_THRESHOLD
        self.confidence_threshold = ZEBRA_CROSSING_CONFIDENCE_THRESHOLD
        
        # ROI settings
        self.roi_start_ratio = ZEBRA_CROSSING_ROI_START_RATIO
        self.roi_end_ratio = ZEBRA_CROSSING_ROI_END_RATIO
        
        # State tracking
        self.zebra_crossing_detected = False
        self.crossing_state = None
        self.state_history = deque(maxlen=ZEBRA_CROSSING_STATE_HISTORY_SIZE)
        self.state_confidence_threshold = ZEBRA_CROSSING_STATE_CONFIDENCE_THRESHOLD
        
        # Statistics
        self.frame_count = 0
        self.crossing_frames = 0
        
    def detect_zebra_crossing(self, frame):
        """
        Detect zebra crossing patterns in frame using HSV color thresholding
        
        Args:
            frame: Input frame from video capture
            
        Returns:
            tuple: (detected, confidence, mask) - whether zebra crossing detected, confidence level, binary mask
        """
        h, w = frame.shape[:2]
        
        # Apply ROI (Region of Interest)
        roi_start = int(h * self.roi_start_ratio)
        roi_end = int(h * self.roi_end_ratio)
        frame_roi = frame[roi_start:roi_end, :]
        
        # Convert to HSV and detect white stripes
        hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # Morphological operations to enhance stripe pattern
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask, frame_roi
    
    def analyze_stripe_pattern(self, mask):
        """
        Analyze horizontal stripe pattern to detect zebra crossing
        
        Args:
            mask: Binary mask of white regions
            
        Returns:
            tuple: (pattern_detected, stripe_count, confidence)
        """
        h, w = mask.shape
        
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
    
    def smooth_state(self, current_detection):
        """
        Apply state smoothing using history buffer to prevent flickering
        
        Args:
            current_detection: Current detection result (boolean)
            
        Returns:
            tuple: (smoothed_state, confidence)
        """
        self.state_history.append(current_detection)
        
        if len(self.state_history) == 0:
            return False, 0.0
        
        # Count detections in history
        detected_count = sum(1 for state in self.state_history if state)
        confidence = detected_count / len(self.state_history)
        
        # Apply confidence threshold
        if confidence >= self.state_confidence_threshold:
            return True, confidence
        else:
            return False, confidence
    
    def get_zebra_crossing_info(self):
        """
        Get current zebra crossing detection state
        
        Returns:
            dict: Current detection information
        """
        return {
            'zebra_crossing_detected': self.zebra_crossing_detected,
            'crossing_state': self.crossing_state,
            'frame_count': self.frame_count,
            'crossing_frames': self.crossing_frames
        }
    
    def run(self, cap, exit_flag):
        """
        Main detection loop for zebra crossing monitoring
        
        Args:
            cap: Shared video capture object
            exit_flag: Dict flag to signal thread exit {'flag': bool}
        """
        retry_count = 0
        max_retries = 100
        
        while not exit_flag['flag']:
            ret, frame = cap.read()
            if not ret:
                retry_count += 1
                if DEBUG_MODE:
                    print(f"[ZEBRA] Failed to read frame. Retry {retry_count}/{max_retries}")
                
                if retry_count >= max_retries:
                    print("[ZEBRA] Max retries reached - exiting thread")
                    break
                continue
            
            retry_count = 0
            self.frame_count += 1
            
            # Detect zebra crossing
            mask, frame_roi = self.detect_zebra_crossing(frame)
            pattern_detected, stripe_count, raw_confidence = self.analyze_stripe_pattern(mask)
            
            # Apply state smoothing
            smoothed_detection, smoothed_confidence = self.smooth_state(pattern_detected)
            
            # Update state
            with self.state_lock:
                self.zebra_crossing_detected = smoothed_detection
                self.crossing_state = "DETECTED" if smoothed_detection else "NOT_DETECTED"
                
                if smoothed_detection:
                    self.crossing_frames += 1
            
            # Debug logging
            if DEBUG_MODE and self.frame_count % LOG_INTERVAL == 0:
                status = "🟩 ZEBRA CROSSING DETECTED" if smoothed_detection else "⬜ No crossing"
                print(f"[ZEBRA] {status} | Stripes: {stripe_count} | Confidence: {smoothed_confidence:.2f}")


