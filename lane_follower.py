import cv2
import numpy as np
import requests
from threading import Lock

class LaneFollower:
    """Lane detection and following module"""
    
    def __init__(self, control_url, state_lock):
        self.control_url = control_url
        self.state_lock = state_lock
        
        # Lane detection parameters
        self.lower_white = np.array([0, 0, 100])
        self.upper_white = np.array([180, 30, 255])
        
        self.last_lane_command = None
        self.red_light_detected = False
        self.person_detected = False
    
    def detect_lane(self, frame):
        """Detect lane lines in frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def find_lane_curvature(self, mask, frame):
        """Calculate lane curvature and offset"""
        h, w = mask.shape
        mask_roi = mask[h//2:, :]
        cols = cv2.findNonZero(mask_roi)
        
        if cols is None or len(cols) < 10:
            return 0, 0
        
        x_coords = cols[:, 0, 0]
        mid_w = w // 2
        left_points = x_coords[x_coords < mid_w]
        right_points = x_coords[x_coords >= mid_w]
        
        left_center = np.mean(left_points) if len(left_points) > 0 else mid_w * 0.25
        right_center = np.mean(right_points) if len(right_points) > 0 else mid_w * 1.75
        
        lane_center = (left_center + right_center) / 2
        image_center = w / 2
        offset = (lane_center - image_center) / image_center
        
        lane_width = right_center - left_center
        curvature = offset * (w / (lane_width + 1e-5))
        
        return curvature, offset
    
    def get_command_by_curvature(self, curvature, offset):
        """Determine movement command based on lane curvature"""
        sharp_left_threshold = -0.5
        gentle_left_threshold = -0.15
        gentle_right_threshold = 0.15
        sharp_right_threshold = 0.5
        
        if curvature < sharp_left_threshold:
            command = "LEFT"
            status = "Sharp Left"
        elif curvature < gentle_left_threshold:
            command = "FORWARD"
            status = "Gentle Left"
        elif curvature > sharp_right_threshold:
            command = "RIGHT"
            status = "Sharp Right"
        elif curvature > gentle_right_threshold:
            command = "FORWARD"
            status = "Gentle Right"
        else:
            command = "FORWARD"
            status = "Go Straight"
        
        return command, status, curvature, offset
    
    def send_command(self, command):
        """Send movement command to vehicle"""
        try:
            response = requests.post(self.control_url, json={'command': command})
            return True
        except Exception as e:
            print("[LANE] Error sending command:", str(e))
            return False
    
    def should_execute(self, red_light_detected, person_detected):
        """Check if lane following should execute based on priority"""
        return not red_light_detected and not person_detected
    
    def get_state(self):
        """Get current lane following state"""
        return {
            'last_lane_command': self.last_lane_command,
            'red_light_detected': self.red_light_detected,
            'person_detected': self.person_detected
        }