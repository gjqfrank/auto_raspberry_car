import os
import cv2
import time
import requests
import torch
import numpy as np
import threading
from threading import Lock
import model.detector
import utils.utils

stream_url = "http://172.20.10.3:8080/?action=stream"
control_url = "http://172.20.10.3:5000/control"

# 全局状态锁
state_lock = Lock()
person_detected = False
person_box = None
last_lane_command = None
exit_flag = False
red_light_detected = False  # ⭐ 红灯标志
traffic_light_state = None  # None, "RED", "GREEN"
light_state_changed = False  # ⭐ 信号灯状态是否改变


class TrafficLightDetector:
    """交通信号灯检测类 - 持续监测信号灯颜色"""
    
    def __init__(self, control_url):
        self.control_url = control_url
        # 红色范围 (HSV)
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])
        
        # 绿色范围 (HSV)
        self.lower_green = np.array([35, 100, 100])
        self.upper_green = np.array([85, 255, 255])
        
        self.last_light_state = None  # 记录上一帧的信号灯状态
    
    def detect_light_color(self, roi):
        """
        分析信号灯区域，判断颜色
        返回: "RED", "GREEN" 或 None
        """
        if roi is None or roi.size == 0:
            return None
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 检测红色
        mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        red_count = cv2.countNonZero(mask_red)
        
        # 检测绿色
        mask_green = cv2.inRange(hsv, self.lower_green, self.upper_green)
        green_count = cv2.countNonZero(mask_green)
        
        total_pixels = roi.shape[0] * roi.shape[1]
        red_ratio = red_count / total_pixels
        green_ratio = green_count / total_pixels
        
        # 判断颜色（比例阈值可调）
        if red_ratio > 0.15:
            return "RED"
        elif green_ratio > 0.15:
            return "GREEN"
        else:
            return None
    
    def extract_roi_from_box(self, frame, box, scale_h, scale_w):
        """从检测框中提取感兴趣区域"""
        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)
        
        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        roi = frame[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)
    
    def is_traffic_light(self, category):
        """检查是否为交通信号灯"""
        traffic_light_names = ["traffic light", "traffic_light"]
        return category.lower() in traffic_light_names
    
    def run_traffic_light_detection(self, cap, LABEL_NAMES, model, cfg):
        """交通信号灯检测线程 - 持续检测和监控"""
        global red_light_detected, traffic_light_state, exit_flag, light_state_changed
        
        while not exit_flag:
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO模型推理
            res_img = cv2.resize(frame, (cfg["width"], cfg["height"]), 
                               interpolation=cv2.INTER_LINEAR)
            img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
            img = torch.from_numpy(img.transpose(0, 3, 1, 2))
            img = img.to("cpu").float() / 255.0
            
            preds = model(img)
            output = utils.utils.handel_preds(preds, cfg, "cpu")
            output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)
            
            h, w, _ = frame.shape
            scale_h, scale_w = h / cfg["height"], w / cfg["width"]
            
            light_found = False
            display_frame = frame.copy()
            current_light_color = None
            
            # 处理检测框，寻找信号灯
            if len(output_boxes[0]) > 0:
                for box in output_boxes[0]:
                    box = box.tolist()
                    obj_score = box[4]
                    category = LABEL_NAMES[int(box[5])]
                    
                    x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
                    x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)
                    
                    if self.is_traffic_light(category):
                        light_found = True
                        
                        # 提取感兴趣区域
                        roi, coords = self.extract_roi_from_box(frame, box, scale_h, scale_w)
                        
                        # 检测颜色
                        light_color = self.detect_light_color(roi)
                        
                        if light_color:
                            current_light_color = light_color
                            
                            # ⭐ 关键：检测状态变化
                            with state_lock:
                                old_state = traffic_light_state
                                traffic_light_state = light_color
                                
                                # 红灯 → 绿灯时触发
                                if old_state == "RED" and light_color == "GREEN":
                                    light_state_changed = True
                                    red_light_detected = False
                                    print(f"\n🟢🟢🟢 [TRAFFIC LIGHT] 红灯变绿灯！准备启动！🟢🟢🟢\n")
                                elif light_color == "RED":
                                    red_light_detected = True
                                    light_state_changed = False
                                else:
                                    red_light_detected = False
                                    light_state_changed = False
                            
                            # 红灯时发送停止命令
                            if light_color == "RED":
                                try:
                                    response = requests.post(self.control_url, json={'command': "STOP"})
                                    print(f"[TRAFFIC LIGHT] 🔴 RED LIGHT - STOP !!! | Score: {obj_score:.2f}")
                                except Exception as e:
                                    print(f"[TRAFFIC LIGHT] Error: {e}")
                            else:
                                print(f"[TRAFFIC LIGHT] 🟢 GREEN LIGHT - Can Move | Score: {obj_score:.2f}")
                            
                            # 绘制检测框和颜色标签
                            color_map = {"RED": (0, 0, 255), "GREEN": (0, 255, 0)}
                            box_color = color_map.get(light_color, (255, 255, 0))
                            
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)
                            cv2.putText(display_frame, f"{category}: {light_color}", 
                                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.8, box_color, 2)
            
            # 如果没有检测到信号灯
            if not light_found:
                with state_lock:
                    traffic_light_state = None
                    red_light_detected = False
                    light_state_changed = False
            
            # 显示状态
            with state_lock:
                status_text = f"Light: {traffic_light_state if traffic_light_state else 'Not Detected'}"
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(display_frame, "[TRAFFIC LIGHT MONITOR]", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.imshow('Traffic Light Detection', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag = True
                break


class LaneFollower:
    """根据赛道曲率跟踪赛道的类"""
    
    def __init__(self, control_url):
        self.control_url = control_url
        # 赛道检测参数
        self.lower_white = np.array([0, 0, 100])      # 白色下界(HSV)
        self.upper_white = np.array([180, 30, 255])   # 白色上界(HSV)
        
    def detect_lane(self, frame):
        """检测赛道线"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def find_lane_curvature(self, mask, frame):
        """计算赛道曲率和偏差"""
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
        """根据曲率返回命令（不直接发送）"""
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
    
    def run_lane_detection(self, cap):
        """赛道检测线程"""
        global person_detected, red_light_detected, last_lane_command, exit_flag
        
        while not exit_flag:
            ret, frame = cap.read()
            if not ret:
                break
            
            h, w, _ = frame.shape
            lane_mask = self.detect_lane(frame)
            curvature, offset = self.find_lane_curvature(lane_mask, frame)
            command, status, curv, off = self.get_command_by_curvature(curvature, offset)
            
            # 优先级检查：红灯 > 人物 > 赛道
            with state_lock:
                is_red_light = red_light_detected
                is_person_detected = person_detected
            
            # 红灯时停止，否则检查人物检测
            if not is_red_light and not is_person_detected:
                try:
                    response = requests.post(self.control_url, json={'command': command})
                    print(f"[LANE] {status} | Cmd: {command} | Curv: {curv:.2f}")
                except Exception as e:
                    print(f"[LANE] Error: {e}")
                last_lane_command = command
            
            # 显示赛道检测的调试信息
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Curvature: {curvature:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Offset: {offset:.2f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            with state_lock:
                if is_red_light:
                    mode_text = "[RED LIGHT]"
                    color = (0, 0, 255)
                elif is_person_detected:
                    mode_text = "[PERSON MODE]"
                    color = (0, 165, 255)
                else:
                    mode_text = "[LANE MODE]"
                    color = (0, 255, 0)
            
            cv2.putText(display_frame, mode_text, (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            display_mask = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([display_frame, display_mask])
            cv2.imshow('Lane Detection | Mask', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag = True
                break


class YoloTracker:
    """YOLO目标检测和跟踪类"""
    
    def __init__(self, control_url, model, cfg):
        self.control_url = control_url
        self.model = model
        self.cfg = cfg
        self.target_categories = ["person"]
    
    def run_yolo_detection(self, cap):
        """YOLO检测线程"""
        global person_detected, red_light_detected, person_box, exit_flag
        
        # 加载label names
        LABEL_NAMES = []
        with open(self.cfg["names"], 'r') as f:
            for line in f.readlines():
                LABEL_NAMES.append(line.strip())
        
        while not exit_flag:
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO模型推理
            res_img = cv2.resize(frame, (self.cfg["width"], self.cfg["height"]), 
                               interpolation=cv2.INTER_LINEAR)
            img = res_img.reshape(1, self.cfg["height"], self.cfg["width"], 3)
            img = torch.from_numpy(img.transpose(0, 3, 1, 2))
            img = img.to("cpu").float() / 255.0
            
            preds = self.model(img)
            output = utils.utils.handel_preds(preds, self.cfg, "cpu")
            output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)
            
            h, w, _ = frame.shape
            scale_h, scale_w = h / self.cfg["height"], w / self.cfg["width"]
            
            person_found = False
            display_frame = frame.copy()
            
            # 处理检测框
            if len(output_boxes[0]) > 0:
                for box in output_boxes[0]:
                    box = box.tolist()
                    obj_score = box[4]
                    category = LABEL_NAMES[int(box[5])]
                    
                    x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
                    x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)
                    
                    # 绘制检测框
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(display_frame, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
                    
                    if category in self.target_categories:
                        person_found = True
                        # 人的跟踪逻辑（优先级低于红灯）
                        
                        with state_lock:
                            is_red_light = red_light_detected
                        
                        if not is_red_light:
                            x_center = (box[0] + box[2]) / 2 / self.cfg["width"]
                            box_area = (box[2] - box[0]) * (box[3] - box[1]) / (self.cfg["height"] * self.cfg["width"])
                            
                            command = None
                            if box_area > 0.7:
                                command = "STOP"
                                action = "STOP - Close"
                            elif x_center < 0.3:
                                command = "LEFT"
                                action = "TURN LEFT"
                            elif x_center > 0.7:
                                command = "RIGHT"
                                action = "TURN RIGHT"
                            else:
                                command = "FORWARD"
                                action = "GO FORWARD"
                            
                            with state_lock:
                                person_detected = True
                                person_box = (x1, y1, x2, y2)
                            
                            # 发送人物跟踪命令
                            try:
                                response = requests.post(self.control_url, json={'command': command})
                                print(f"[YOLO] {action} | Area: {box_area:.3f} | Center: {x_center:.2f}")
                            except Exception as e:
                                print(f"[YOLO] Error: {e}")
            
            # 如果没有检测到人
            if not person_found:
                with state_lock:
                    person_detected = False
                    person_box = None
            
            # 显示状态
            with state_lock:
                status = "Person Tracking" if person_detected else "No Person"
                is_red_light = red_light_detected
            
            status_color = (0, 0, 255) if is_red_light else (0, 255, 0)
            cv2.putText(display_frame, f"Status: {status}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(display_frame, "[YOLO MODE]", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.imshow('YOLO Detection', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag = True
                break


def main():
    """主程序"""
    global exit_flag
    
    print("=" * 70)
    print("初始化三模式小车控制系统 (信号灯 + 人物 + 赛道)")
    print("=" * 70)
    
    # 加载配置和模型
    print("\n加载模型...")
    cfg = utils.utils.load_datafile('data/coco.data')
    weights = 'modelzoo/coco2017-0.241078ap-model.pth'
    assert os.path.exists(weights), "请指定正确的模型路径"
    
    device = "cpu"
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    print("✓ 模型加载完成")
    
    print("\n打开视频流...")
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("✗ 无法打开视频流")
        return
    print("✓ 视频流打开成功")
    
    # 加载label names
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())
    
    # 创建实例
    traffic_light_detector = TrafficLightDetector(control_url)
    lane_follower = LaneFollower(control_url)
    yolo_tracker = YoloTracker(control_url, model, cfg)
    
    print("\n启动三线程...")
    print("-" * 70)
    print("优先级 1 (最高): 交通信号灯模式 - 持续监测")
    print("  ├─ 🔴 红灯 → STOP（红灯时停止）")
    print("  └─ 🟢 绿灯 → CONTINUE（绿灯时继续）")
    print("优先级 2 (中等): 人物检测模式 - 检测到人时接管控制")
    print("优先级 3 (最低): 赛道跟踪模式 - 沿着赛道行驶")
    print("-" * 70)
    print("按 'q' 键退出程序\n")
    
    # 创建线程
    traffic_thread = threading.Thread(target=traffic_light_detector.run_traffic_light_detection, 
                                     args=(cap, LABEL_NAMES, model, cfg), daemon=True)
    lane_thread = threading.Thread(target=lane_follower.run_lane_detection, args=(cap,), daemon=True)
    yolo_thread = threading.Thread(target=yolo_tracker.run_yolo_detection, args=(cap,), daemon=True)
    
    # 启动线程
    traffic_thread.start()
    lane_thread.start()
    yolo_thread.start()
    
    # 等待线程结束
    try:
        traffic_thread.join()
        lane_thread.join()
        yolo_thread.join()
    except KeyboardInterrupt:
        print("\n程序被中断")
        exit_flag = True
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ 程序正常结束")


if __name__ == '__main__':
    main()