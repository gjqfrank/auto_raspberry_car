import os
import cv2
import time
import argparse
import requests
import torch
import numpy as np
import model.detector
import utils.utils

stream_url = "http://172.20.10.3:8080/?action=stream"    # 树莓派 IP 地址
control_url = "http://172.20.10.3:5000/control"          # 树莓派控制 URL
cap = cv2.VideoCapture(stream_url)

cfg = utils.utils.load_datafile('data/coco.data')
weights = 'modelzoo/coco2017-0.241078ap-model.pth'
assert os.path.exists(weights), "请指定正确的模型路径"

target_categories = ["person"]

device = "cpu"

model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
model.load_state_dict(torch.load(weights, map_location=device))
model.eval()

class LaneFollower:
    """根据赛道曲率跟踪赛道的类"""
    
    def __init__(self, control_url):
        self.control_url = control_url
        # 赛道检测参数
        self.lower_white = np.array([0, 0, 100])      # 白色下界(HSV)
        self.upper_white = np.array([180, 30, 255])   # 白色上界(HSV)
        
    def detect_lane(self, frame):
        """检测赛道线"""
        # 转换到HSV色彩空间(更适合色彩检测)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 根据白色范围创建掩膜
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # 形态学操作，去除噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def find_lane_curvature(self, mask, frame):
        """
        通过分析掩膜来计算赛道曲率
        返回: (曲率值, 偏差量)
        曲率为负表示向左弯，正值表示向右弯
        """
        h, w = mask.shape
        
        # 只关注图像下半部分(车前方)
        mask_roi = mask[h//2:, :]
        
        # 获取白色线的列坐标
        cols = cv2.findNonZero(mask_roi)
        
        if cols is None or len(cols) < 10:
            return 0, 0  # 未检测到赛道，返回0
        
        # 提取x坐标
        x_coords = cols[:, 0, 0]
        
        # 计算左右两侧的重心
        mid_w = w // 2
        left_points = x_coords[x_coords < mid_w]
        right_points = x_coords[x_coords >= mid_w]
        
        left_center = np.mean(left_points) if len(left_points) > 0 else mid_w * 0.25
        right_center = np.mean(right_points) if len(right_points) > 0 else mid_w * 1.75
        
        # 计算赛道中心与图像中心的偏差
        lane_center = (left_center + right_center) / 2
        image_center = w / 2
        offset = (lane_center - image_center) / image_center  # 归一化偏差 [-1, 1]
        
        # 计算曲率: 使用赛道宽度的变化来表示弯曲程度
        lane_width = right_center - left_center
        curvature = offset * (w / (lane_width + 1e-5))  # 避免除以0
        
        return curvature, offset
    
    def send_command_by_curvature(self, curvature, offset, frame_h, frame_w):
        """
        根据曲率和偏差发送控制命令
        
        参数说明:
        - curvature: 赛道曲率(-∞ 到 +∞)，负值向左，正值向右
        - offset: 中心偏差 [-1, 1]
        - frame_h, frame_w: 帧的高度和宽度
        """
        
        # 定义控制阈值
        sharp_left_threshold = -0.5      # 急左转阈值
        gentle_left_threshold = -0.15    # 缓左转阈值
        gentle_right_threshold = 0.15    # 缓右转阈值
        sharp_right_threshold = 0.5      # 急右转阈值
        
        print(f"Curvature: {curvature:.3f}, Offset: {offset:.3f}")
        
        command = None
        
        # 根据曲率决定命令
        if curvature < sharp_left_threshold:
            # 急剧左转
            command = "LEFT"
            print("Sharp Left Turn")
        elif curvature < gentle_left_threshold:
            # 缓和左转 - 这里可以前进但倾向左
            command = "FORWARD"  # 也可以自定义为 "GENTLE_LEFT"
            print("Gentle Left Turn")
        elif curvature > sharp_right_threshold:
            # 急剧右转
            command = "RIGHT"
            print("Sharp Right Turn")
        elif curvature > gentle_right_threshold:
            # 缓和右转
            command = "FORWARD"  # 也可以自定义为 "GENTLE_RIGHT"
            print("Gentle Right Turn")
        else:
            # 直线行驶
            command = "FORWARD"
            print("Go Straight")
        
        # 发送命令到树莓派
        try:
            response = requests.post(self.control_url, json={'command': command})
            print(f"Command sent: {response.status_code}")
        except Exception as e:
            print(f"Error sending command: {e}")
    
    def run(self):
        """主循环"""
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            h, w, _ = frame.shape
            
            # 1. 检测赛道线
            lane_mask = self.detect_lane(frame)
            
            # 2. 计算赛道曲率和偏差
            curvature, offset = self.find_lane_curvature(lane_mask, frame)
            
            # 3. 根据曲率发送控制命令
            self.send_command_by_curvature(curvature, offset, h, w)
            
            # 4. 绘制调试信息
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Curvature: {curvature:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Offset: {offset:.2f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示掩膜和原图并排
            display_mask = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([display_frame, display_mask])
            cv2.imshow('Lane Following - Original | Mask', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    follower = LaneFollower(control_url)
    follower.run()