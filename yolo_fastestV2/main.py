import cv2
import numpy as np
import requests
import time

stream_url = "http://172.20.10.5:8080/?action=stream"
control_url = "http://172.20.10.5:5000/control"

cap = cv2.VideoCapture(stream_url)

last_command = "STOP"

def send_command(cmd):
    global last_command
    if cmd != last_command:   # ❗避免重复刷指令（很重要）
        try:
            requests.post(control_url, json={'command': cmd}, timeout=0.2)
            print("Send:", cmd)
            last_command = cmd
        except:
            pass


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # =========================
    # 1. 转 HSV（核心）
    # =========================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # =========================
    # 2. 红色检测（两个区间）
    # =========================
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = mask1 + mask2

    # =========================
    # 3. 绿色检测
    # =========================
    lower_green = np.array([40, 70, 70])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # =========================
    # 4. 计算面积（判断是否“看到灯”）
    # =========================
    red_area = cv2.countNonZero(red_mask)
    green_area = cv2.countNonZero(green_mask)

    print("RED:", red_area, "GREEN:", green_area)

    # =========================
    # 5. 决策逻辑（核心）
    # =========================
    if red_area > 2000:
        send_command("STOP")
        cv2.putText(frame, "RED LIGHT - STOP", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    elif green_area > 2000:
        send_command("FORWARD")
        cv2.putText(frame, "GREEN LIGHT - GO", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    else:
        send_command("STOP")
        cv2.putText(frame, "NO LIGHT - STOP", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    # =========================
    # 6. 显示调试画面
    # =========================
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
