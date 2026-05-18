# ...existing code...
"""
Multi-mode Vehicle Control System - Main Entry Point (UPDATED)
===============================================================

Orchestrates multiple detection modules:
- Traffic Light Detection (COLOR-BASED)
- Zebra Crossing Detection (NEW)
- Lane Following (with zebra crossing awareness)

⚠️  DISABLED: Person Safety Detection

All parameters are imported from constants.py.

Author: Auto Vehicle Control System
Date: 2026-05-18
"""

import os
import cv2
import torch
import threading
from threading import Lock, Event
import model.detector
import utils.utils
from traffic_light_detector import TrafficLightDetector
from zebra_crossing_detector import ZebraCrossingDetector
from lane_follower import LaneFollower
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
    ENABLE_PERSON_DETECTION,
    ENABLE_ZEBRA_CROSSING_DETECTION,
    ENABLE_TRAFFIC_LIGHT_DETECTION,
    ENABLE_LANE_FOLLOWING,
)

# ============================================================================
# Global State Management
# ============================================================================
state_lock = Lock()
exit_flag = {'flag': False}  # Use dict to allow modification in threads


class TrafficLightThread:
    """Thread for traffic light detection (COLOR-BASED, no YOLO model needed)"""
    
    def __init__(self, state_lock):
        self.state_lock = state_lock
        self.detector = TrafficLightDetector(CONTROL_URL, state_lock)
    
    def run(self, cap, LABEL_NAMES, exit_flag):
        """Run traffic light detection in thread"""
        self.detector.run(cap, LABEL_NAMES, exit_flag)


class ZebraCrossingThread:
    """Thread for zebra crossing detection (NEW)"""
    
    def __init__(self, state_lock):
        self.state_lock = state_lock
        self.detector = ZebraCrossingDetector(CONTROL_URL, state_lock)
    
    def run(self, cap, exit_flag):
        """Run zebra crossing detection in thread"""
        self.detector.run(cap, exit_flag)


class LaneFollowingThread:
    """Thread for lane detection and following"""
    
    def __init__(self, state_lock):
        self.state_lock = state_lock
        self.follower = LaneFollower(CONTROL_URL, state_lock)
        self.cfg = utils.utils.load_datafile(CONFIG_FILE)
    
    def run(self, cap, exit_flag):
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
                self.follower.zebra_crossing_detected = getattr(self, 'zebra_crossing_detected', False)
            
            # Check priority: red_light > zebra_crossing > lane_following
            # When zebra crossing detected, lane follower continues (safe passage)
            if self.follower.should_execute(self.follower.red_light_detected, 
                                           self.follower.zebra_crossing_detected):
                self.follower.send_command(command)
                if DEBUG_MODE:
                    mode_info = ""
                    if self.follower.zebra_crossing_detected:
                        mode_info = " [ZEBRA CROSSING]"
                    print(f"[LANE]{mode_info} {status} | Cmd: {command} | Curv: {curv:.2f}")
                self.follower.last_lane_command = command
            
            if SHOW_VISUAL_OUTPUT:
                try:
                    display_frame = frame.copy()
                    cv2.putText(display_frame, f"Curvature: {curvature:.2f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Offset: {offset:.2f}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if self.follower.red_light_detected:
                        mode_text = "[RED LIGHT - STOP]"
                        color = (0, 0, 255)
                    elif self.follower.zebra_crossing_detected:
                        mode_text = "[ZEBRA CROSSING - CONTINUE]"
                        color = (255, 0, 0)
                    else:
                        mode_text = "[LANE MODE]"
                        color = (0, 255, 0)
                    
                    cv2.putText(display_frame, mode_text, (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    display_mask = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
                    combined = cv2.hstack([display_frame, display_mask])
                    cv2.imshow('Lane Detection | Mask', combined)
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"[LANE] Warning: Cannot display window - {str(e)}")


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
    print("=" * 90)
    print("🚗 Multi-Mode Vehicle Control System (UPDATED)")
    print("=" * 90)
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
    print(f"\n🎮 Active Detection Modules:")
    print(f"   ✅ Traffic Light Detection (COLOR-BASED)" if ENABLE_TRAFFIC_LIGHT_DETECTION else "   ❌ Traffic Light Detection")
    print(f"   ✅ Zebra Crossing Detection (NEW)" if ENABLE_ZEBRA_CROSSING_DETECTION else "   ❌ Zebra Crossing Detection")
    print(f"   ✅ Lane Following" if ENABLE_LANE_FOLLOWING else "   ❌ Lane Following")
    print(f"   ❌ Person Detection (DISABLED)" if not ENABLE_PERSON_DETECTION else "   ✅ Person Detection")
    print(f"\n🎯 Updated Priority System (Person Detection Disabled):")
    print(f"   ┌─ Level 1 (Highest): 🛣️  Zebra Crossing Detection")
    print(f"   │   └─ When zebra crossing detected: CONTINUE lane following (safe passage)")
    print(f"   ├─ Level 2 (High): 🚦 Traffic Light Detection (Color-Based)")
    print(f"   │   ├─ RED detected: Stop vehicle")
    print(f"   │   └─ GREEN detected: Continue driving")
    print(f"   └─ Level 3 (Lowest): 🛣️  Lane Following Mode")
    print(f"\n📋 All parameters can be modified in: constants.py")
    print(f"🔧 Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}")
    print(f"🖼️  Visual Output: {'ON' if SHOW_VISUAL_OUTPUT else 'OFF'}")
    print("=" * 90)
    print("Press 'q' to exit the program\n")


def main():
    """Main entry point for vehicle control system"""
    
    print_system_info()
    
    # ========================================================================
    # Check Feature Switches
    # ========================================================================
    if not ENABLE_PERSON_DETECTION:
        print("ℹ️  Person detection is DISABLED in constants.py")
    
    if not ENABLE_ZEBRA_CROSSING_DETECTION:
        print("⚠️  Zebra crossing detection is DISABLED in constants.py")
    
    # ========================================================================
    # Load YOLO Model (still used for general object detection if needed)
    # ========================================================================
    print("⏳ Loading model...")
    cfg = utils.utils.load_datafile(CONFIG_FILE)
    assert os.path.exists(WEIGHTS_PATH), f"❌ Model weights not found: {WEIGHTS_PATH}"
    
    device = DEVICE
    detector_model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    detector_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    detector_model.eval()
    print("✅ Model loaded successfully")
    
    print("\n📌 Detection Methods:")
    print("   🚦 Traffic Light: COLOR-BASED (no YOLO)")
    print("   🛣️  Zebra Crossing: COLOR-BASED (no YOLO)")
    print("   🛣️  Lane Following: COLOR-BASED (no YOLO)")
    print("   ❌ Person Detection: DISABLED")
    
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
    # Initialize Detection Threads
    # ========================================================================
    print("\n⏳ Starting detection threads...")
    print("-" * 90)
    
    threads = []
    
    # Traffic Light Detection Thread
    if ENABLE_TRAFFIC_LIGHT_DETECTION:
        traffic_thread_obj = TrafficLightThread(state_lock)
        traffic_thread = threading.Thread(
            target=traffic_thread_obj.run, 
            args=(shared_cap, LABEL_NAMES, exit_flag), 
            daemon=True,
            name="TrafficLightDetector"
        )
        traffic_thread.start()
        threads.append(traffic_thread)
        print(f"✅ Traffic Light Detection Thread: STARTED (COLOR-BASED)")
    else:
        print(f"⏭️  Traffic Light Detection Thread: SKIPPED (disabled)")
    
    # Zebra Crossing Detection Thread (NEW)
    if ENABLE_ZEBRA_CROSSING_DETECTION:
        zebra_thread_obj = ZebraCrossingThread(state_lock)
        zebra_thread = threading.Thread(
            target=zebra_thread_obj.run, 
            args=(shared_cap, exit_flag), 
            daemon=True,
            name="ZebraCrossingDetector"
        )
        zebra_thread.start()
        threads.append(zebra_thread)
        print(f"✅ Zebra Crossing Detection Thread: STARTED (NEW)")
    else:
        print(f"⏭️  Zebra Crossing Detection Thread: SKIPPED (disabled)")
    
    # Lane Following Thread
    if ENABLE_LANE_FOLLOWING:
        lane_thread_obj = LaneFollowingThread(state_lock)
        lane_thread = threading.Thread(
            target=lane_thread_obj.run, 
            args=(shared_cap, exit_flag), 
            daemon=True,
            name="LaneFollower"
        )
        lane_thread.start()
        threads.append(lane_thread)
        print(f"✅ Lane Following Thread: STARTED")
    else:
        print(f"⏭️  Lane Following Thread: SKIPPED (disabled)")
    
    # Person Detection Thread (DISABLED by default)
    if ENABLE_PERSON_DETECTION:
        print(f"⚠️  Person Safety Detection: NOT IMPLEMENTED (disabled)")
    else:
        print(f"❌ Person Safety Detection: DISABLED")
    
    print("-" * 90)
    print("\n🚗 System is now running. Press 'q' to exit.\n")
    
    # ========================================================================
    # Main Display Loop
    # ========================================================================
    try:
        while not exit_flag['flag']:
            ret, frame = shared_cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            mode_text = "[MULTI-MODE DETECTION]"
            color = (0, 255, 0)
            
            cv2.putText(frame, mode_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            processed_frame = frame
            
            cv2.imshow('Processed Frame', processed_frame)
    
    except KeyboardInterrupt:
        print("\n⚠️  Program interrupted by user (Ctrl+C)")
        exit_flag['flag'] = True
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        exit_flag['flag'] = True
    
    # ========================================================================
    # Wait for Threads to Complete
    # ========================================================================
    print("\n⏳ Shutting down detection threads...")
    try:
        for thread in threads:
            thread.join(timeout=5)
    except Exception as e:
        if DEBUG_MODE:
            print(f"[MAIN] Error joining threads: {str(e)}")
    
    # ========================================================================
    # Cleanup Resources
    # ========================================================================
    finally:
        exit_flag['flag'] = True
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
