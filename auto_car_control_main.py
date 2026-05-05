import os
import cv2
import torch
import threading
from threading import Lock
import model.detector
import utils.utils
from traffic_light_detector import TrafficLightDetector
from person_detector import PersonDetector
from lane_follower import LaneFollower

stream_url = "http://172.20.10.3:8080/?action=stream"
control_url = "http://172.20.10.3:5000/control"

state_lock = Lock()
exit_flag = False


class TrafficLightThread:
    """Thread for traffic light detection"""
    
    def __init__(self, state_lock):
        self.state_lock = state_lock
        self.detector = TrafficLightDetector(control_url, state_lock)
    
    def run(self, cap, LABEL_NAMES):
        global exit_flag
        self.detector.run(cap, LABEL_NAMES)


class PersonDetectionThread:
    """Thread for person detection"""
    
    def __init__(self, state_lock):
        self.state_lock = state_lock
        self.detector = PersonDetector(control_url, state_lock)
    
    def run(self, cap, LABEL_NAMES):
        global exit_flag
        self.detector.run(cap, LABEL_NAMES)


class LaneFollowingThread:
    """Thread for lane detection and following"""
    
    def __init__(self, state_lock):
        self.state_lock = state_lock
        self.follower = LaneFollower(control_url, state_lock)
        self.cfg = utils.utils.load_datafile('data/coco.data')
    
    def run(self, cap):
        global exit_flag
        
        while not exit_flag:
            ret, frame = cap.read()
            if not ret:
                break
            
            h, w, _ = frame.shape
            lane_mask = self.follower.detect_lane(frame)
            curvature, offset = self.follower.find_lane_curvature(lane_mask, frame)
            command, status, curv, off = self.follower.get_command_by_curvature(curvature, offset)
            
            with self.state_lock:
                self.follower.red_light_detected = getattr(self, 'red_light_detected', False)
                self.follower.person_detected = getattr(self, 'person_detected', False)
            
            if self.follower.should_execute(self.follower.red_light_detected, self.follower.person_detected):
                self.follower.send_command(command)
                print(f"[LANE] {status} | Cmd: {command} | Curv: {curv:.2f}")
                self.follower.last_lane_command = command
            
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Curvature: {curvature:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Offset: {offset:.2f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if self.follower.red_light_detected:
                mode_text = "[RED LIGHT]"
                color = (0, 0, 255)
            elif self.follower.person_detected:
                mode_text = "[PERSON DETECTED]"
                color = (0, 0, 255)
            else:
                mode_text = "[LANE MODE]"
                color = (0, 255, 0)
            
            cv2.putText(display_frame, mode_text, (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            display_mask = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
            combined = cv2.hstack([display_frame, display_mask])
            cv2.imshow('Lane Detection | Mask', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag = True
                break


def main():
    global exit_flag
    
    print("=" * 70)
    print("Multi-mode vehicle control system initialization")
    print("=" * 70)
    
    print("\nLoading model...")
    cfg = utils.utils.load_datafile('data/coco.data')
    weights = 'modelzoo/coco2017-0.241078ap-model.pth'
    assert os.path.exists(weights), "Please specify correct model path"
    
    device = "cpu"
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    print("Model loaded successfully")
    
    print("\nOpening video stream...")
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Failed to open video stream")
        return
    print("Video stream opened successfully")
    
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())
    
    print("\nStarting three threads...")
    print("-" * 70)
    print("Priority 1 (Highest): Traffic Light Mode - Continuous monitoring")
    print("  - RED LIGHT: Stop car")
    print("  - GREEN LIGHT: Continue driving")
    print("Priority 2 (Medium): Person Detection Mode")
    print("  - Person detected: Stop car")
    print("  - No person: Continue lane following")
    print("Priority 3 (Lowest): Lane Following Mode")
    print("-" * 70)
    print("Press 'q' to exit\n")
    
    traffic_thread_obj = TrafficLightThread(state_lock)
    person_thread_obj = PersonDetectionThread(state_lock)
    lane_thread_obj = LaneFollowingThread(state_lock)
    
    traffic_thread = threading.Thread(target=traffic_thread_obj.run, args=(cap, LABEL_NAMES), daemon=True)
    person_thread = threading.Thread(target=person_thread_obj.run, args=(cap, LABEL_NAMES), daemon=True)
    lane_thread = threading.Thread(target=lane_thread_obj.run, args=(cap,), daemon=True)
    
    traffic_thread.start()
    person_thread.start()
    lane_thread.start()
    
    try:
        traffic_thread.join()
        person_thread.join()
        lane_thread.join()
    except KeyboardInterrupt:
        print("\nProgram interrupted")
        exit_flag = True
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nProgram ended successfully")


if __name__ == '__main__':
    main()