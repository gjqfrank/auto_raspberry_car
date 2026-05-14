# ...existing code...
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
from threading import Lock, Event
import model.detector
import utils.utils
from traffic_light_detector import TrafficLightDetector
from lane_follower import LaneFollower
from person_safety_detector import PersonSafetyDetector
import time

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

# ============================================================================
# Shared Display State - for aggregated visualization
# ============================================================================
display_state = {
    'traffic_frame': None,
    'safety_frame': None,
    'lane_frame': None,
    'status_text': '',
    'fps': 0.0,
    'lock': Lock()
}


class TrafficLightThread:
    """Thread for traffic light detection"""
    
    def __init__(self, state_lock):
        self.state_lock = state_lock
        self.detector = TrafficLightDetector(CONTROL_URL, state_lock)
    
    def run(self, cap, LABEL_NAMES, exit_flag):
        """Run traffic light detection in thread"""
        self.detector.run(cap, LABEL_NAMES, exit_flag)


class PersonSafetyThread:
    """Thread for person safety distance monitoring"""
    
    def __init__(self, state_lock, cfg, detector_model, device):
        self.state_lock = state_lock
        self.cfg = cfg
        self.detector_model = detector_model
        self.device = device
        self.detector = PersonSafetyDetector(CONTROL_URL, state_lock)
    
    def run(self, cap, LABEL_NAMES, exit_flag):
        """Run person safety detection in thread"""
        self.detector.run(cap, self.cfg, self.detector_model, self.device, LABEL_NAMES, exit_flag)


class LaneFollowingThread:
    """Thread for lane detection and following"""
    
    def __init__(self, state_lock):
        self.state_lock = state_lock
        self.follower = LaneFollower(CONTROL_URL, state_lock)
        self.cfg = utils.utils.load_datafile(CONFIG_FILE)
    
    def run(self, cap, exit_flag, display_state):
        retry_count = 0
        max_retries = 100
        """Run lane following detection in thread"""
        
        while not exit_flag['flag']:
            ret, frame = cap.read()
            if not ret:
                retry_count += 1
                if DEBUG_MODE:
                    print(f"[LANE] Failed to read frame. Retry {retry_count}/{max_retries}")
            
                if retry_count >= max_retries:
                    print("[LANE] Max retries reached - exiting thread")
                    break
                time.sleep(1)
                continue
            
            
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
            
            # ================================================================
            # Visualization - Lane Detection Output
            # ================================================================
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
            
            # Update display state for aggregated visualization
            with display_state['lock']:
                display_frame_resized = cv2.resize(display_frame, (320, 240))
                display_state['lane_frame'] = display_frame_resized
                display_state['status_text'] = f"Lane: {status} | {mode_text}"
            
            if SHOW_VISUAL_OUTPUT:
                try:
                    display_mask = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
                    combined = cv2.hstack([display_frame, display_mask])
                    cv2.imshow('Lane Detection | Mask', combined)
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"[LANE] Warning: Cannot display window - {str(e)}")
            
            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if DEBUG_MODE:
                    print("[LANE] 'q' key detected - initiating shutdown")
                exit_flag['flag'] = True
                break


# ---------------------------------------------------------------------------
# Shared frame provider - read once and share to all detectors
# ---------------------------------------------------------------------------
class SharedFrameCapture:
    """A lightweight proxy that provides .read() to consumers using the latest frame."""
    def __init__(self):
        self.lock = Lock()
        self.event = Event()
        self.frame = None
        self.ret = False
        self.stopped = False

    def update_frame(self, ret, frame):
        with self.lock:
            self.ret = ret
            # store a copy to avoid race conditions when OpenCV reuses buffers
            self.frame = None if frame is None else frame.copy()
            # notify consumers a new frame is available
            self.event.set()

    def read(self):
        """Return the most recent frame. Wait briefly if none available yet."""
        # wait until at least one frame is available or stop requested
        if not self.event.wait(timeout=1.0):
            # timeout - no frame available
            return False, None
        with self.lock:
            ret = self.ret
            frame = None if self.frame is None else self.frame.copy()
        # clear event so next wait will block until a new frame arrives
        self.event.clear()
        return ret, frame

    def stop(self):
        self.stopped = True
        self.event.set()


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


def create_display_canvas(display_state):
    """
    Create an aggregated display canvas showing all detection results.
    Mimics rasp_yolo.py visualization approach.
    """
    try:
        with display_state['lock']:
            lane_frame = display_state['lane_frame']
            status_text = display_state['status_text']
            fps = display_state['fps']
        
        # Create canvas with proper size for multiple streams
        canvas_height = 480
        canvas_width = 640
        canvas = cv2.Mat(canvas_height, canvas_width, cv2.CV_8UC3)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        canvas[:] = (30, 30, 30)  # Dark background
        
        # Display lane frame if available
        if lane_frame is not None:
            h, w = lane_frame.shape[:2]
            y_offset = (canvas_height - h) // 2
            x_offset = (canvas_width - w) // 2
            canvas[y_offset:y_offset+h, x_offset:x_offset+w] = lane_frame
        
        # Add status information
        cv2.putText(canvas, status_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(canvas, f"FPS: {fps:.1f}", (20, canvas_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(canvas, "Press 'q' to exit", (canvas_width - 280, canvas_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return canvas
    except Exception as e:
        if DEBUG_MODE:
            print(f"[DISPLAY] Error creating canvas: {str(e)}")
        return None


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
    # Open Video Stream (single reader) and create shared frame provider
    # ========================================================================
    print("\n⏳ Opening video stream...")
    real_cap = cv2.VideoCapture(STREAM_URL)
    if not real_cap.isOpened():
        print("❌ Failed to open video stream")
        print(f"   Check if Raspberry Pi is running at: {STREAM_URL}")
        return
    print(f"✅ Video stream opened: {STREAM_URL}")

    # Create shared frame capture and a reader thread that updates it
    shared_cap = SharedFrameCapture()

    def frame_reader_loop(real_cap, shared_cap, exit_flag):
        retry_count = 0
        while not exit_flag['flag']:
            ret, frame = real_cap.read()
            if not ret:
                retry_count += 1
                if DEBUG_MODE:
                    print(f"[FRAME READER] Failed to read frame. Retry {retry_count}")
                time.sleep(0.1)
                continue
            retry_count = 0
            shared_cap.update_frame(ret, frame)
        # signal consumers to stop
        shared_cap.stop()

    frame_reader = threading.Thread(
        target=frame_reader_loop,
        args=(real_cap, shared_cap, exit_flag),
        daemon=True,
        name="FrameReader"
    )
    frame_reader.start()
    
    # ========================================================================
    # Load Label Names
    # ========================================================================
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())
    print(f"✅ Loaded {len(LABEL_NAMES)} class labels")
    
    # ========================================================================
    # Initialize Detection Threads (pass shared_cap instead of real cap)
    # ========================================================================
    print("\n⏳ Starting detection threads...")
    print("-" * 80)
    
    traffic_thread_obj = TrafficLightThread(state_lock)
    safety_thread_obj = PersonSafetyThread(state_lock, cfg, detector_model, device)
    lane_thread_obj = LaneFollowingThread(state_lock)
    
    traffic_thread = threading.Thread(
        target=traffic_thread_obj.run, 
        args=(shared_cap, LABEL_NAMES, exit_flag), 
        daemon=True,
        name="TrafficLightDetector"
    )
    safety_thread = threading.Thread(
        target=safety_thread_obj.run, 
        args=(shared_cap, LABEL_NAMES, exit_flag), 
        daemon=True,
        name="PersonSafetyDetector"
    )
    lane_thread = threading.Thread(
        target=lane_thread_obj.run, 
        args=(shared_cap, exit_flag, display_state), 
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
    # Display Loop - Aggregate and show all detection results
    # ========================================================================
    frame_count = 0
    start_time = time.time()
    
    try:
        while not exit_flag['flag']:
            # Create aggregated display
            display_canvas = create_display_canvas(display_state)
            
            if display_canvas is not None:
                cv2.imshow('🚗 Multi-Mode Vehicle Control - Real-time Visualization', display_canvas)
            
            # Calculate and update FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                with display_state['lock']:
                    display_state['fps'] = fps
                frame_count = 0
                start_time = time.time()
            
            # Check for 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if DEBUG_MODE:
                    print("[MAIN] 'q' key detected - initiating shutdown")
                exit_flag['flag'] = True
                break
            
            time.sleep(0.033)  # ~30 FPS for display loop
    
    except KeyboardInterrupt:
        print("\n⚠️  Program interrupted by user (Ctrl+C)")
        exit_flag['flag'] = True
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        exit_flag['flag'] = True
    
    # ========================================================================
    # Wait for Threads to Complete
    # ========================================================================
    try:
        traffic_thread.join(timeout=5)
        safety_thread.join(timeout=5)
        lane_thread.join(timeout=5)
    except Exception as e:
        if DEBUG_MODE:
            print(f"[MAIN] Error joining threads: {str(e)}")
    
    # ========================================================================
    # Cleanup Resources
    # ========================================================================
    finally:
        exit_flag['flag'] = True
        # stop frame reader and release real capture
        shared_cap.stop()
        if frame_reader.is_alive():
            frame_reader.join(timeout=1.0)
        real_cap.release()
        cv2.destroyAllWindows()
        print("\n✅ All resources released")
        print("✅ Program ended successfully")


if __name__ == '__main__':
    main()
# ...existing code...