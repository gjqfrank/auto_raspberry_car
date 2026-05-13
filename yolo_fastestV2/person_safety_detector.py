import cv2
import torch
import numpy as np
import requests
import time
from threading import Lock
import model.detector
import utils.utils
from collections import deque


class PersonSafetyDetector:
    """
    Person safety distance detector module
    
    Stops vehicle when person is detected within danger zone
    Resumes normal operation when person moves beyond safe distance
    """
    
    def __init__(self, control_url, state_lock):
        self.control_url = control_url
        self.state_lock = state_lock
        
        # Safety parameters (as percentage of frame area)
        self.DANGER_DISTANCE_THRESHOLD = 0.50      # 50% of frame = too close, STOP
        self.SAFE_DISTANCE_THRESHOLD = 0.35        # 35% of frame = safe distance
        
        # State tracking
        self.person_in_danger_zone = False
        self.person_detected = False
        self.current_person_distance = 0.0
        self.distance_history = deque(maxlen=5)    # Smooth distance readings
        
        # Performance monitoring
        self.last_command_time = 0
        self.min_command_interval = 0.05  # Minimum 50ms between commands
        self.last_command = None
        
        # Debug info
        self.frame_count = 0
        self.danger_zone_frames = 0
        
    def calculate_person_distance_ratio(self, box, frame_height, frame_width, model_height):
        """
        Calculate the relative distance to person based on detection box area
        
        Args:
            box: Detection box [x1, y1, x2, y2, conf, cls] in model coordinates
            frame_height: Original frame height
            frame_width: Original frame width
            model_height: Model input height
            
        Returns:
            distance_ratio: 0-1 value where higher = closer to camera
        """
        # Calculate scale factors
        scale_h = frame_height / model_height
        model_width = model_height * 16 // 9  # Assume 16:9 aspect ratio
        scale_w = frame_width / model_width
        
        # Convert box coordinates to pixel area
        box_width = (box[2] - box[0]) * scale_w
        box_height = (box[3] - box[1]) * scale_h
        box_area = box_width * box_height
        
        # Calculate frame area
        frame_area = frame_height * frame_width
        
        # Return distance ratio
        distance_ratio = box_area / frame_area
        return distance_ratio
    
    def smooth_distance(self, raw_distance):
        """
        Apply smoothing to distance measurements to reduce jitter
        
        Args:
            raw_distance: Raw distance ratio from detection
            
        Returns:
            smoothed_distance: Smoothed distance value
        """
        self.distance_history.append(raw_distance)
        if len(self.distance_history) > 0:
            return np.mean(self.distance_history)
        return raw_distance
    
    def send_command(self, command):
        """
        Send control command to vehicle with rate limiting
        
        Args:
            command: 'STOP' or other control commands
            
        Returns:
            bool: True if command sent successfully
        """
        current_time = time.time()
        
        # Rate limiting - avoid sending commands too frequently
        if current_time - self.last_command_time < self.min_command_interval:
            return False
        
        # Avoid sending duplicate commands
        if command == self.last_command:
            return False
        
        try:
            response = requests.post(self.control_url, json={'command': command}, timeout=1)
            self.last_command = command
            self.last_command_time = current_time
            return response.status_code == 200
        except Exception as e:
            print(f"[SAFETY] Error sending command: {str(e)}")
            return False
    
    def run(self, cap, cfg, model, device, LABEL_NAMES):
        """
        Main detection loop for person safety monitoring
        
        Args:
            cap: Video capture object
            cfg: Model configuration
            model: YOLO detector model
            device: Torch device
            LABEL_NAMES: List of class names
        """
        global exit_flag
        from auto_car_control_main import exit_flag as global_exit_flag
        
        print("[SAFETY] Person safety detector started")
        print(f"[SAFETY] Danger zone threshold: {self.DANGER_DISTANCE_THRESHOLD*100:.0f}%")
        print(f"[SAFETY] Safe distance threshold: {self.SAFE_DISTANCE_THRESHOLD*100:.0f}%")
        
        target_categories = ["person"]
        
        while not global_exit_flag:
            ret, frame = cap.read()
            if not ret:
                print("[SAFETY] Failed to read frame")
                break
            
            self.frame_count += 1
            h, w = frame.shape[:2]
            
            # Run YOLO detection
            res_img = cv2.resize(frame, (cfg["width"], cfg["height"]), 
                                interpolation=cv2.INTER_LINEAR)
            img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
            img = torch.from_numpy(img.transpose(0, 3, 1, 2))
            img = img.to(device).float() / 255.0
            
            # Inference
            with torch.no_grad():
                preds = model(img)
            
            # Post-processing
            output = utils.utils.handel_preds(preds, cfg, device)
            output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)
            
            # Process detections
            self.person_detected = False
            max_distance = 0.0
            scale_h, scale_w = h / cfg["height"], w / cfg["width"]
            
            for box in output_boxes[0]:
                box = box.tolist()
                obj_score = box[4]
                category = LABEL_NAMES[int(box[5])]
                
                if category in target_categories:
                    self.person_detected = True
                    
                    # Calculate distance to person
                    distance = self.calculate_person_distance_ratio(
                        box, h, w, cfg["height"]
                    )
                    max_distance = max(max_distance, distance)
            
            # Update state with smoothed distance
            if self.person_detected:
                self.current_person_distance = self.smooth_distance(max_distance)
            else:
                self.current_person_distance = self.smooth_distance(0)
            
            # ========== STATE MACHINE ==========
            
            # STATE: Person enters danger zone
            if self.person_detected and self.current_person_distance > self.DANGER_DISTANCE_THRESHOLD:
                if not self.person_in_danger_zone:
                    self.person_in_danger_zone = True
                    print(f"[SAFETY] ALERT! Person in danger zone - Distance: {self.current_person_distance:.2%}")
                    self.send_command("STOP")
                self.danger_zone_frames += 1
            
            # STATE: Person leaves danger zone
            elif self.person_in_danger_zone and self.current_person_distance < self.SAFE_DISTANCE_THRESHOLD:
                self.person_in_danger_zone = False
                print(f"[SAFETY] Person moved to safe distance - Distance: {self.current_person_distance:.2%}")
                # Note: Don't send resume command here - let other modules decide
                # Resume is controlled by the priority system in main thread
            
            # STATE: No person detected - clear danger flag
            elif not self.person_detected:
                if self.person_in_danger_zone:
                    print("[SAFETY] Person no longer detected")
                self.person_in_danger_zone = False
                self.distance_history.clear()
            
            # Update shared state for other modules
            with self.state_lock:
                self.person_in_danger_zone = self.person_in_danger_zone
            
            # Logging every 30 frames
            if self.frame_count % 30 == 0:
                status = "DANGER ZONE" if self.person_in_danger_zone else "SAFE"
                print(f"[SAFETY] Frame {self.frame_count} | Person: {self.person_detected} | "
                      f"Distance: {self.current_person_distance:.2%} | Status: {status}")
            
            # Allow key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def get_state(self):
        """
        Get current safety detector state
        
        Returns:
            dict: Current state information
        """
        return {
            'person_in_danger_zone': self.person_in_danger_zone,
            'person_detected': self.person_detected,
            'current_distance': self.current_person_distance,
            'danger_zone_frames': self.danger_zone_frames,
            'total_frames': self.frame_count
        }
    
    def is_person_safe(self):
        """
        Check if person is at safe distance
        
        Returns:
            bool: True if person is safe (not in danger zone)
        """
        return not self.person_in_danger_zone
