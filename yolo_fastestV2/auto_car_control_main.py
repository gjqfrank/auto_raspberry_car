# ...existing code...
"""
Multi-mode Vehicle Control System - Main Entry Point (UPDATED)
===============================================================

Orchestrates multiple detection modules:
- Traffic Light Detection (COLOR-BASED - SIMPLIFIED)
- Zebra Crossing Detection
- Lane Following (with zebra crossing awareness)
- COMPREHENSIVE VISUALIZATION with annotations

⚠️  DISABLED: Person Safety Detection

All parameters are imported from constants.py.

Author: Auto Vehicle Control System
Date: 2026-05-26
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
import numpy as np

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
    ENABLE_VISUALIZATION,
    LANE_ROI_START_RATIO,
    ZEBRA_CROSSING_ROI_START_RATIO,
    ZEBRA_CROSSING_ROI_END_RATIO,
    CAMERA_BRIGHTNESS_GAIN,
)

# ============================================================================
# Global State Management
# ============================================================================
state_lock = Lock()
exit_flag = {'flag': False}  # Use dict to allow modification in threads

# 全局状态共享变量
global_traffic_light_state = {'red': False, 'green': False, 'yellow': False}
global_zebra_crossing_state = {'detected': False, 'mask': None}
global_lane_state = {'mask': None, 'curvature': 0, 'offset': 0, 'radius': 0}


class TrafficLightThread:
    """Thread for traffic light detection (COLOR-BASED - SIMPLIFIED)"""
    
    def __init__(self, state_lock):
        self.state_lock = state_lock
        self.detector = TrafficLightDetector(CONTROL_URL, state_lock)
    
    def run(self, cap, LABEL_NAMES, exit_flag):
        """Run traffic light detection in thread"""
        self.detector.run(cap, LABEL_NAMES, exit_flag)


class ZebraCrossingThread:
    """Thread for zebra crossing detection"""
    
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
            lane_mask, lane_mask_roi, lane_roi_y = self.follower.detect_lane(frame)
            curvature, offset, radius, left_x, right_x = self.follower.find_lane_curvature(lane_mask_roi, frame, lane_roi_y)
            command, status, curv, off = self.follower.get_command_by_curvature(curvature, offset)
            
            with self.state_lock:
                self.follower.red_light_detected = getattr(self, 'red_light_detected', False)
                self.follower.zebra_crossing_detected = getattr(self, 'zebra_crossing_detected', False)
                
                # Update global state for visualization
                global_lane_state['mask'] = lane_mask
                global_lane_state['curvature'] = curvature
                global_lane_state['offset'] = offset
                global_lane_state['radius'] = radius
            
            # Check priority: red_light > zebra_crossing > lane_following
            # When zebra crossing detected, lane follower continues (safe passage)
            if self.follower.should_execute(self.follower.red_light_detected, 
                                           self.follower.zebra_crossing_detected):
                self.follower.send_command(command)
                if DEBUG_MODE:
                    mode_info = ""
                    if self.follower.zebra_crossing_detected:
                        mode_info = " [ZEBRA CROSSING]"
                    print(f"[LANE]{mode_info} {status} | Cmd: {command} | Curv: {curv:.2f} | Radius: {radius:.2f}")
                self.follower.last_lane_command = command


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


def draw_lane_annotations(frame, lane_mask, curvature, offset, radius, roi_y_start):
    """
    在完整画面上标注车道线、曲率和曲率半径
    
    Args:
        frame: 原始画面
        lane_mask: 车道检测掩码
        curvature: 车道曲率
        offset: 车道偏移
        radius: 曲率半径
        roi_y_start: ROI起始Y坐标
    
    Returns:
        标注后的画面
    """
    h, w = frame.shape[:2]
    display_frame = frame.copy()
    
    # 绘制ROI区域边界 (绿色边界表示检测区域)
    roi_height = int(h * (1 - LANE_ROI_START_RATIO))
    cv2.rectangle(display_frame, (0, roi_y_start), (w, h), (0, 255, 0), 2)
    cv2.putText(display_frame, "Lane Detection ROI (Bottom 1/3)", (10, roi_y_start - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 从掩码中提取车道线轮廓
    if lane_mask is not None:
        # 找到车道线的轮廓
        contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 左右车道线分离
        left_contours = []
        right_contours = []
        
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                if cx < w // 2:
                    left_contours.append(contour)
                else:
                    right_contours.append(contour)
        
        # 绘制左车道线 (蓝色)
        if left_contours:
            largest_left = max(left_contours, key=cv2.contourArea)
            cv2.drawContours(display_frame, [largest_left], 0, (255, 0, 0), 3)
            cv2.putText(display_frame, "Left Lane", (20, roi_y_start + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 绘制右车道线 (红色)
        if right_contours:
            largest_right = max(right_contours, key=cv2.contourArea)
            cv2.drawContours(display_frame, [largest_right], 0, (0, 0, 255), 3)
            cv2.putText(display_frame, "Right Lane", (w - 200, roi_y_start + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # 绘制中心线
    center_x = int(w / 2 + offset * w / 2)
    cv2.line(display_frame, (center_x, roi_y_start), (center_x, h), (255, 255, 0), 2)
    
    # 显示统计信息
    info_y = 30
    cv2.putText(display_frame, f"Curvature: {curvature:.3f}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Offset: {offset:.3f}", (10, info_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 显示曲率半径 (如果半径有效)
    if radius > 0 and radius != float('inf'):
        cv2.putText(display_frame, f"Radius: {radius:.1f} pixels", (10, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "Radius: Linear", (10, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return display_frame


def draw_zebra_crossing_annotations(frame, zebra_mask, detected, roi_y_start):
    """
    在完整画面上标注斑马线检测区域和结果
    
    Args:
        frame: 原始画面
        zebra_mask: 斑马线检测掩码 (ROI大小)
        detected: 是否检测到斑马线
        roi_y_start: ROI起始Y坐标
    
    Returns:
        标注后的画面
    """
    display_frame = frame.copy()
    h, w = frame.shape[:2]
    
    # 绘制ROI区域边界
    cv2.rectangle(display_frame, (0, roi_y_start), (w, h), (255, 165, 0), 2)
    
    # 标注斑马线检测状态
    if detected:
        status_text = "🟨 ZEBRA CROSSING DETECTED"
        color = (0, 255, 255)
        cv2.putText(display_frame, status_text, (10, roi_y_start - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 在检测区域绘制掩码 (FIXED: Handle ROI mask shape correctly)
        if zebra_mask is not None and zebra_mask.size > 0:
            # zebra_mask 是 ROI 大小，需要正确处理 
            mask_alpha = zebra_mask > 0
            if mask_alpha.any():
                # 创建黄色显示图像（只在有掩码的区域）
                zebra_display = np.zeros_like(zebra_mask, dtype=np.uint8)
                zebra_display[mask_alpha] = 255
                
                # 将 ROI 掩码转换为 BGR
                zebra_display_bgr = cv2.cvtColor(zebra_display, cv2.COLOR_GRAY2BGR)
                zebra_display_bgr[mask_alpha] = (0, 255, 255)  # 黄色
                
                # 获取 ROI 区域并进行加权融合
                roi_region = display_frame[roi_y_start:h, :]
                result = cv2.addWeighted(roi_region[mask_alpha], 0.7, 
                                        zebra_display_bgr[mask_alpha], 0.3, 0)
                roi_region[mask_alpha] = result
    else:
        cv2.putText(display_frame, "Zebra Crossing: NOT DETECTED", (10, roi_y_start - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    
    return display_frame


def draw_traffic_light_annotations(frame, red_detected, green_detected, yellow_detected):
    """
    在完整画面上标注红绿灯检测结果
    
    Args:
        frame: 原始画面
        red_detected: 是否检测到红灯
        green_detected: 是否检测到绿灯
        yellow_detected: 是否检测到黄灯
    
    Returns:
        标注后的画面
    """
    display_frame = frame.copy()
    h, w = frame.shape[:2]
    
    # 右上角显示红绿灯状态
    status_box_y = 30
    
    if red_detected:
        status_text = "🔴 RED LIGHT"
        color = (0, 0, 255)
        command_text = "STATUS: STOP"
    elif green_detected:
        status_text = "🟢 GREEN LIGHT"
        color = (0, 255, 0)
        command_text = "STATUS: GO"
    elif yellow_detected:
        status_text = "🟡 YELLOW LIGHT"
        color = (0, 255, 255)
        command_text = "STATUS: CAUTION"
    else:
        status_text = "⚪ NO LIGHT"
        color = (200, 200, 200)
        command_text = "STATUS: UNKNOWN"
    
    cv2.putText(display_frame, status_text, (w - 300, status_box_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(display_frame, command_text, (w - 300, status_box_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return display_frame


def print_system_info():
    """Print system information and configuration"""
    print("=" * 90)
    print("🚗 Multi-Mode Vehicle Control System (SIMPLIFIED TRAFFIC LIGHT)")
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
    print(f"   ✅ Traffic Light Detection (COLOR-BASED - SIMPLIFIED)" if ENABLE_TRAFFIC_LIGHT_DETECTION else "   ❌ Traffic Light Detection")
    print(f"   ✅ Zebra Crossing Detection" if ENABLE_ZEBRA_CROSSING_DETECTION else "   ❌ Zebra Crossing Detection")
    print(f"   ✅ Lane Following" if ENABLE_LANE_FOLLOWING else "   ❌ Lane Following")
    print(f"   ❌ Person Detection (DISABLED)" if not ENABLE_PERSON_DETECTION else "   ✅ Person Detection")
    print(f"\n🎨 Visualization:")
    print(f"   ✅ COMPREHENSIVE ANNOTATIONS ENABLED" if ENABLE_VISUALIZATION else "   ❌ Visualization disabled")
    print(f"   • Lane detection areas with left/right lane markers")
    print(f"   • Curvature and radius of curvature")
    print(f"   • Zebra crossing detection areas")
    print(f"   • Traffic light status")
    print(f"\n🎯 Updated Priority System (Person Detection Disabled):")
    print(f"   ┌─ Level 1 (Highest): 🛣️  Zebra Crossing Detection")
    print(f"   │   └─ When zebra crossing detected: CONTINUE lane following (safe passage)")
    print(f"   ├─ Level 2 (High): 🚦 Traffic Light Detection (SIMPLIFIED - Direct Area Detection)")
    print(f"   │   ├─ RED area > 2000px: Stop vehicle")
    print(f"   │   └─ GREEN area > 2000px: Continue driving")
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
    print("   🚦 Traffic Light: COLOR-BASED - SIMPLIFIED (no YOLO, no state smoothing)")
    print("   🛣️  Zebra Crossing: COLOR-BASED (no YOLO)")
    print("   🛣️  Lane Following: COLOR-BASED (no YOLO) - Bottom 1/3 only")
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

            if CAMERA_BRIGHTNESS_GAIN != 1.0:
                frame = np.clip(frame * CAMERA_BRIGHTNESS_GAIN, 0, 255).astype(np.uint8)

            shared_cap.update_frame(ret, frame)
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
    lane_thread_obj = None
    zebra_thread_obj = None
    traffic_thread_obj = None
    
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
        print(f"✅ Traffic Light Detection Thread: STARTED (COLOR-BASED - SIMPLIFIED)")
    else:
        print(f"⏭️  Traffic Light Detection Thread: SKIPPED (disabled)")
    
    # Zebra Crossing Detection Thread
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
        print(f"✅ Zebra Crossing Detection Thread: STARTED")
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
        print(f"✅ Lane Following Thread: STARTED (Bottom 1/3 Detection)")
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
    # Main Display Loop - WITH COMPREHENSIVE VISUALIZATION
    # ========================================================================
    try:
        while not exit_flag['flag']:
            ret, frame = shared_cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            display_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # 添加标题
            cv2.putText(display_frame, "[COMPREHENSIVE DETECTION & ANNOTATION]", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 绘制车道线标注
            if ENABLE_LANE_FOLLOWING and lane_thread_obj:
                lane_roi_y = int(h * LANE_ROI_START_RATIO)
                lane_mask = global_lane_state['mask']
                curvature = global_lane_state['curvature']
                offset = global_lane_state['offset']
                radius = global_lane_state['radius']
                
                if lane_mask is not None:
                    display_frame = draw_lane_annotations(
                        display_frame, lane_mask, curvature, offset, radius, lane_roi_y
                    )
            
            # 绘制斑马线标注
            if ENABLE_ZEBRA_CROSSING_DETECTION and zebra_thread_obj:
                zebra_roi_y = int(h * ZEBRA_CROSSING_ROI_START_RATIO)
                zebra_detected = zebra_thread_obj.detector.zebra_crossing_detected
                zebra_mask = zebra_thread_obj.detector.zebra_mask
                
                display_frame = draw_zebra_crossing_annotations(
                    display_frame, zebra_mask, zebra_detected, zebra_roi_y
                )
            
            # 绘制红绿灯标注
            if ENABLE_TRAFFIC_LIGHT_DETECTION and traffic_thread_obj:
                red_detected = traffic_thread_obj.detector.red_light_detected
                green_detected = traffic_thread_obj.detector.traffic_light_state == "GREEN"
                yellow_detected = traffic_thread_obj.detector.traffic_light_state == "YELLOW"
                
                display_frame = draw_traffic_light_annotations(
                    display_frame, red_detected, green_detected, yellow_detected
                )
            
            cv2.imshow('🚗 Auto Car Control - Full Frame with Annotations', display_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag['flag'] = True
                break
    
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
