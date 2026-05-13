import os
import cv2
import requests
import torch
import model.detector
import utils.utils

class PersonDetector:
    """Person detection module - detects if person is visible, stops when person detected"""
    
    def __init__(self, control_url, state_lock):
        self.control_url = control_url
        self.state_lock = state_lock
        self.target_categories = ["person"]
        
        self.cfg = utils.utils.load_datafile('data/coco.data')
        self.weights = 'modelzoo/coco2017-0.241078ap-model.pth'
        assert os.path.exists(self.weights), "Please specify correct model path"
        
        self.device = "cpu"
        self.model = model.detector.Detector(self.cfg["classes"], self.cfg["anchor_num"], True).to(self.device)
        self.model.load_state_dict(torch.load(self.weights, map_location=self.device))
        self.model.eval()
        
        self.person_detected = False
        self.person_box = None
        self.red_light_detected = False
    
    def send_stop_command(self):
        """Send stop command when person is detected"""
        try:
            response = requests.post(self.control_url, json={'command': "STOP"})
            print("[PERSON] Person detected - Stop")
            return True
        except Exception as e:
            print("[PERSON] Error sending command:", str(e))
            return False
    
    def run(self, cap, LABEL_NAMES):
        """Main detection loop - continuously check if person is visible"""
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
            
            person_found = False
            
            if len(output_boxes[0]) > 0:
                for box in output_boxes[0]:
                    box = box.tolist()
                    
                    obj_score = box[4]
                    category = LABEL_NAMES[int(box[5])]
                    
                    x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
                    x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)
                    
                    if category in self.target_categories:
                        person_found = True
                        
                        with self.state_lock:
                            self.red_light_detected = getattr(self, 'red_light_detected', False)
                        
                        if not self.red_light_detected:
                            self.send_stop_command()
                        
                        with self.state_lock:
                            self.person_detected = True
                            self.person_box = (x1, y1, x2, y2)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, category, (x1, y1 - 25), 0, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
            
            if not person_found:
                with self.state_lock:
                    self.person_detected = False
                    self.person_box = None
                print("[PERSON] No person detected - Can continue")
            
            cv2.imshow('Person Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def get_state(self):
        """Get current person detection state"""
        return {
            'person_detected': self.person_detected,
            'person_box': self.person_box
        }