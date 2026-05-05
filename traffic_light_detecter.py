import os
import cv2
import requests
import torch
import numpy as np
import model.detector
import utils.utils

class TrafficLightDetector:
    """Traffic light detection and control module"""
    
    def __init__(self, control_url, state_lock):
        self.control_url = control_url
        self.state_lock = state_lock
        
        self.cfg = utils.utils.load_datafile('data/coco.data')
        self.weights = 'modelzoo/coco2017-0.241078ap-model.pth'
        assert os.path.exists(self.weights), "Please specify correct model path"
        
        self.device = "cpu"
        self.model = model.detector.Detector(self.cfg["classes"], self.cfg["anchor_num"], True).to(self.device)
        self.model.load_state_dict(torch.load(self.weights, map_location=self.device))
        self.model.eval()
        
        # Red color range (HSV)
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])
        
        # Green color range (HSV)
        self.lower_green = np.array([35, 100, 100])
        self.upper_green = np.array([85, 255, 255])
        
        self.last_light_state = None
        self.red_light_detected = False
        self.traffic_light_state = None
        self.light_state_changed = False
    
    def detect_light_color(self, roi):
        """Detect traffic light color in ROI region"""
        if roi is None or roi.size == 0:
            return None
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        red_count = cv2.countNonZero(mask_red)
        
        mask_green = cv2.inRange(hsv, self.lower_green, self.upper_green)
        green_count = cv2.countNonZero(mask_green)
        
        total_pixels = roi.shape[0] * roi.shape[1]
        red_ratio = red_count / total_pixels
        green_ratio = green_count / total_pixels
        
        if red_ratio > 0.15:
            return "RED"
        elif green_ratio > 0.15:
            return "GREEN"
        else:
            return None
    
    def extract_roi_from_box(self, frame, box, scale_h, scale_w):
        """Extract region of interest from detection box"""
        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        roi = frame[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)
    
    def is_traffic_light(self, category):
        """Check if detected object is traffic light"""
        return category.lower() in ["traffic light", "traffic_light"]
    
    def detect_state_change(self, current_color):
        """Detect if traffic light state changed from RED to GREEN"""
        old_state = self.traffic_light_state
        self.traffic_light_state = current_color
        
        if old_state == "RED" and current_color == "GREEN":
            self.light_state_changed = True
            return True
        
        self.light_state_changed = False
        return False
    
    def get_command_by_light_color(self, light_color):
        """Determine movement command based on traffic light color"""
        if light_color == "RED":
            command = "STOP"
            status = "Red light"
        elif light_color == "GREEN":
            command = "CONTINUE"
            status = "Green light"
        else:
            command = "STOP"
            status = "Unknown light"
        
        return command, status
    
    def send_command(self, command):
        """Send movement command to vehicle"""
        try:
            response = requests.post(self.control_url, json={'command': command})
            return True
        except Exception as e:
            print("[TRAFFIC LIGHT] Error sending command:", str(e))
            return False
    
    def should_execute(self, light_color):
        """Check if traffic light command should be executed"""
        return light_color is not None
    
    def run(self, cap, LABEL_NAMES):
        """Main detection loop"""
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            res_img = cv2.resize(frame, (self.cfg["width"], self.cfg["height"]), 
                               interpolation=cv2.INTER_LINEAR)
            img = res_img.reshape(1, self.cfg["height"], self.cfg["width"], 3)
            img = torch.from_numpy(img.transpose(0, 3, 1, 2))
            img = img.to(self.device).float() / 255.0
            
            preds = self.model(img)
            output = utils.utils.handel_preds(preds, self.cfg, self.device)
            output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)
            
            h, w, _ = frame.shape
            scale_h, scale_w = h / self.cfg["height"], w / self.cfg["width"]
            
            if len(output_boxes[0]) == 0:
                response = requests.post(self.control_url, json={'command': "STOP"})
                print("[TRAFFIC LIGHT] No traffic light detected - Stop")
            
            for box in output_boxes[0]:
                box = box.tolist()
                
                obj_score = box[4]
                category = LABEL_NAMES[int(box[5])]
                
                x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
                x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)
                
                if self.is_traffic_light(category):
                    roi, coords = self.extract_roi_from_box(frame, box, scale_h, scale_w)
                    light_color = self.detect_light_color(roi)
                    
                    if light_color:
                        state_changed = self.detect_state_change(light_color)
                        
                        with self.state_lock:
                            if light_color == "RED":
                                self.red_light_detected = True
                            else:
                                self.red_light_detected = False
                        
                        command, status = self.get_command_by_light_color(light_color)
                        self.send_command(command)
                        print(f"[TRAFFIC LIGHT] {status}")
                        
                        if state_changed:
                            print("[TRAFFIC LIGHT] Signal changed from RED to GREEN - Ready to move")
                        
                        color_map = {"RED": (0, 0, 255), "GREEN": (0, 255, 0)}
                        box_color = color_map.get(light_color, (255, 255, 0))
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                        cv2.putText(frame, f"{category}: {light_color}", (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(frame, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
                
                cv2.putText(frame, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Traffic Light Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def get_state(self):
        """Get current traffic light state"""
        return {
            'red_light_detected': self.red_light_detected,
            'traffic_light_state': self.traffic_light_state,
            'light_state_changed': self.light_state_changed
        }