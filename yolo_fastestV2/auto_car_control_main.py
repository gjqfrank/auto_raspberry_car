"""
Multi-mode Vehicle Control System - Main Entry Point
=====================================================

Orchestrates multiple detection modules (Traffic Light, Person Safety, Lane Following)
with proper priority system and thread management.

Imports all parameters from constants.py - modify there to adjust behavior.

Author: Auto Vehicle Control System
Date: 2026-05-13
"""

import os
import cv2
import torch
import threading
from threading import Lock
import model.detector
import utils.utils
from traffic_light_detector import TrafficLightDetector
from lane_follower import LaneFollower
from person_safety_detector import PersonSafetyDetector

# ============================================================================
# Import all constants
# ============================================================================
from constants import (
    STREAM_URL,
    CONTROL_URL,
    CONFIG_FILE,
    WEIGHTS_PATH,
    DEVICE,
    NMS_CONF_THRESHOLD,
    NMS_IOU_THRESHOLD,
    DEBUG_MODE,
    LOG_INTERVAL,
    SHOW_VISUAL_OUTPUT,
    DANGER_DISTANCE_THRESHOLD,
    SAFE_DISTANCE_THRESHOLD,
)

# ============================================================================
# Global State Management
# ============================================================================
state_lock = Lock()
exit_flag = {'flag': False}  # Use dict to allow modification in threads


class TrafficLightThread:
    """Thread for traffic light detection"""
    
    def __init__(self, state_lock):
        self.state_lock = state_lock
        self.detector = TrafficLightDetector(CONTROL_URL, state_lock)
    
    def run(self, cap, LABEL_NAMES):
        """Run traffic light detection in thread"""
        self.detector.run(cap, LABEL_NAMES)


class PersonSafetyThread:
    """Thread for person safety distance monitoring"""
    
    def __init__(self, state_lock, cfg, detector_model, device):
        self.state_lock = state_lock
        self.cfg = cfg
        self.detector_model = detector_model
        self.device = device
        self.detector = PersonSafetyDetector(CONTROL_URL, state_lock)
    
    def run(self, cap, LABEL_NAMES):
        """Run person safety detection in thread"""
        self.detector.run(cap, self.cfg, self.detector_model, self.device, LABEL_NAMES)


class LaneFollowingThread:
    """Thread for lane detection and following"""
    
    def __init__(self, state_lock):
        self.state_lock = state_lock
        self.follower = LaneFollower(CONTROL_URL, state_lock)
        self.cfg = utils.utils.load_datafile(CONFIG_FILE)
    
    def run(self, cap):
        """Run lane following detection in thread"""
        
        while not exit_flag['flag']:
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
                # Get person safety status from shared state
                self.follower.person_in_danger_zone = getattr(self, 'person_in_danger_zone', False)
            
            # Check priority: red light > person safety > person detection > lane following
            if self.follower.should_execute(self.follower.red_light_detected, 
                                           self.follower.person_detected,
                                           getattr(self.follower, 'person_in_danger_zone', False)):
                self.follower.send_command(command)
                if DEBUG_MODE:
                    print(f"[LANE] {status} | Cmd: {command} | Curv: {curv:.2f}")
                self.follower.last_lane_command = command
            
            if SHOW_VISUAL_OUTPUT:
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Curvature: {curvature:.2f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Offset: {offset:.2f}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if self.follower.red_light_detected:
                    mode_text = "[RED LIGHT - STOP]"
                    color = (0, 0, 255)
                elif getattr(self.follower, 'person_in_danger_zone', False):
                    mode_text = "[PERSON TOO CLOSE - EMERGENCY STOP]"
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
                exit_flag['flag'] = True
                break


def print_system_info():
    """Print system information and configuration"""
    print("=" * 80)
    print("🚗 Multi-Mode Vehicle Control System with Safety Detection")
    print("=" * 80)
    print(f"\n📡 Network Configuration:")
    print(f"   Stream URL: {STREAM_URL}")
    print(f"   Control URL: {CONTROL_URL}")
    print(f"\n🤖 Model Configuration:")
    print(f"   Config File: {CONFIG_FILE}")
    print(f"   Weights: {WEIGHTS_PATH}")
    print(f"   Device: {DEVICE}")
    print(f"\n⚙️  Inference Parameters:")
    print(f"   Confidence Threshold: {NMS_CONF_THRESHOLD}")
    print(f"   IoU Threshold: {NMS_IOU_THRESHOLD}")
    print(f"\n🛡️  Safety Thresholds:")
    print(f"   Person Danger Zone: {DANGER_DISTANCE_THRESHOLD*100:.0f}% of frame")
    print(f"   Person Safe Distance: {SAFE_DISTANCE_THRESHOLD*100:.0f}% of frame")
    print(f"\n🎮 Control Modes (Priority Order):")
    print(f"   ┌─ Level 1 (Highest): Person Safety Distance Monitoring")
    print(f"   │   └─ If person in danger zone (>{DANGER_DISTANCE_THRESHOLD*100:.0f}%): EMERGENCY STOP")
    print(f"   ├─ Level 2 (High): Traffic Light Detection")
    print(f"   │   ├─ RED LIGHT: Stop vehicle")
    print(f"   │   └─ GREEN LIGHT: Continue driving")
    print(f"   ├─ Level 3 (Medium): Person Detection")
    print(f"   │   ├─ Person detected: Stop vehicle")
    print(f"   │   └─ No person: Lane following mode")
    print(f"   └─ Level 4 (Lowest): Lane Following Mode")
    print(f"\n📋 All parameters can be modified in: constants.py")
    print(f"🔧 Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}")
    print(f"🖼️  Visual Output: {'ON' if SHOW_VISUAL_OUTPUT else 'OFF'}")
    print("=" * 80)
    print("Press 'q' to exit the program\n")


def main():
    """Main entry point for vehicle control system"""
    
    print_system_info()
    
    # ========================================================================
    # Load YOLO Model
    # ========================================================================
    print("⏳ Loading YOLO model...")
    cfg = utils.utils.load_datafile(CONFIG_FILE)
    assert os.path.exists(WEIGHTS_PATH), f"❌ Model weights not found: {WEIGHTS_PATH}"
    
    device = DEVICE
    detector_model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    detector_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    detector_model.eval()
    print("✅ Model loaded successfully")
    
    # ========================================================================
    # Open Video Stream
    # ========================================================================
    print("\n⏳ Opening video stream...")
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("❌ Failed to open video stream")
        print(f"   Check if Raspberry Pi is running at: {STREAM_URL}")
        return
    print(f"✅ Video stream opened: {STREAM_URL}")
    
    # ========================================================================
    # Load Label Names
    # ========================================================================
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())
    print(f"✅ Loaded {len(LABEL_NAMES)} class labels")
    
    # ========================================================================
    # Initialize Detection Threads
    # ========================================================================
    print("\n⏳ Starting detection threads...")
    print("-" * 80)
    
    traffic_thread_obj = TrafficLightThread(state_lock)
    safety_thread_obj = PersonSafetyThread(state_lock, cfg, detector_model, device)
    lane_thread_obj = LaneFollowingThread(state_lock)
    
    traffic_thread = threading.Thread(
        target=traffic_thread_obj.run, 
        args=(cap, LABEL_NAMES), 
        daemon=True,
        name="TrafficLightDetector"
    )
    safety_thread = threading.Thread(
        target=safety_thread_obj.run, 
        args=(cap, LABEL_NAMES), 
        daemon=True,
        name="PersonSafetyDetector"
    )
    lane_thread = threading.Thread(
        target=lane_thread_obj.run, 
        args=(cap,), 
        daemon=True,
        name="LaneFollower"
    )
    
    # ========================================================================
    # Start All Threads
    # ========================================================================
    traffic_thread.start()
    safety_thread.start()
    lane_thread.start()
    
    print(f"✅ Traffic Light Detection Thread: STARTED")
    print(f"✅ Person Safety Detection Thread: STARTED")
    print(f"✅ Lane Following Thread: STARTED")
    print("-" * 80)
    print("\n🚗 System is now running. Press 'q' to exit.\n")
    
    # ========================================================================
    # Wait for Threads to Complete
    # ========================================================================
    try:
        traffic_thread.join()
        safety_thread.join()
        lane_thread.join()
    except KeyboardInterrupt:
        print("\n⚠️  Program interrupted by user (Ctrl+C)")
        exit_flag['flag'] = True
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        exit_flag['flag'] = True
    finally:
        # ====================================================================
        # Cleanup Resources
        # ====================================================================
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ All resources released")
        print("✅ Program ended successfully")


if __name__ == '__main__':
    main()
