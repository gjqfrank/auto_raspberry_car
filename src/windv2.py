import cv2
import numpy as np
import math
import time
import threading
import requests
from dataclasses import dataclass

try:
    import pigpio
    PIGPIO_AVAILABLE = True
except ImportError:
    PIGPIO_AVAILABLE = False

try:
    import ncnn
    NCNN_AVAILABLE = True
except ImportError:
    NCNN_AVAILABLE = False

LF_FWD = 1
LF_BWD = 7
LF_PWM = 12
LB_FWD = 24
LB_BWD = 23
LB_PWM = 18
RF_FWD = 6
RF_BWD = 5
RF_PWM = 13
RB_FWD = 21
RB_BWD = 20
RB_PWM = 19

PWM_FREQ = 100
DUTY_STRAIGHT = 200000
DUTY_SLOW = 170000
DUTY_LEFT_STR = 330000
DUTY_RIGHT_STR = 170000

last_offset = 0

IMG_WIDTH = 320
IMG_HEIGHT = 240

CANNY_LOW_THRESHOLD = 60
CANNY_HIGH_THRESHOLD = 120

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 35
HOUGH_MIN_LINE_LENGTH = 20
HOUGH_MAX_LINE_GAP = 40

PERSON_CLASS_ID = 1
BICYCLE_CLASS_ID = 2
CAR_CLASS_ID = 3
MOTORBIKE_CLASS_ID = 4
TRAFFIC_LIGHT_CLASS_ID = 10
STOP_SIGN_CLASS_ID = 12

CLASS_NAMES = [
    "background", "person", "bicycle",
    "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
    "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

INTERESTED_CLASSES = {PERSON_CLASS_ID, BICYCLE_CLASS_ID, CAR_CLASS_ID,
                     MOTORBIKE_CLASS_ID, TRAFFIC_LIGHT_CLASS_ID, STOP_SIGN_CLASS_ID}


@dataclass
class TargetBox:
    x1: int
    y1: int
    x2: int
    y2: int
    cate: int
    score: float

    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class YoloFastestV2:
    def __init__(self):
        self.net = None
        self.input_width = 320
        self.input_height = 320

    def load_model(self, param_path, bin_path):
        if not NCNN_AVAILABLE:
            print("ncnn Python库未安装，YOLO检测不可用")
            return False
        self.net = ncnn.Net()
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        return True

    def detection(self, img, thresh=0.3):
        if self.net is None:
            return []

        h, w = img.shape[:2]
        mat = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.Pixel.PIXEL_BGR2RGB,
                                          w, h, self.input_width, self.input_height)
        mean_vals = [0.0, 0.0, 0.0]
        norm_vals = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
        mat.substract_mean_normalize(mean_vals, norm_vals)

        ex = self.net.create_extractor()
        ex.input("input", mat)
        ret1, out1 = ex.extract("output1")
        ret2, out2 = ex.extract("output2")

        boxes = []
        return boxes


class MotorControl:
    def __init__(self, use_pigpio=True, control_url=None):
        self.pi = None
        self.use_pigpio = use_pigpio and PIGPIO_AVAILABLE
        self.control_url = control_url

        if self.use_pigpio:
            self.pi = pigpio.pi()
            if self.pi is None:
                print("pigpio连接失败！")
                self.use_pigpio = False
            else:
                self._init_pins()

    def _init_pins(self):
        pins_out = [LF_FWD, LF_BWD, LB_FWD, LB_BWD,
                    RF_FWD, RF_BWD, RB_FWD, RB_BWD,
                    LF_PWM, LB_PWM, RF_PWM, RB_PWM]
        for pin in pins_out:
            self.pi.set_mode(pin, pigpio.OUTPUT)

    def _set_pwm(self, pin, freq, duty):
        if self.use_pigpio:
            self.pi.hardware_PWM(pin, freq, duty)

    def _write(self, pin, value):
        if self.use_pigpio:
            self.pi.write(pin, value)

    def _send_command(self, command):
        if self.control_url:
            try:
                requests.post(self.control_url, json={'command': command}, timeout=1)
            except Exception as e:
                print(f"Connection error: {e}")

    def stop(self):
        for pin in [LF_FWD, LF_BWD, LB_FWD, LB_BWD, RF_FWD, RF_BWD, RB_FWD, RB_BWD]:
            self._write(pin, 0)
        for pin in [LF_PWM, LB_PWM, RF_PWM, RB_PWM]:
            self._set_pwm(pin, 0, 0)
        self._send_command("STOP")

    def forward_slow(self):
        for fwd, bwd in [(LF_FWD, LF_BWD), (LB_FWD, LB_BWD),
                         (RF_FWD, RF_BWD), (RB_FWD, RB_BWD)]:
            self._write(fwd, 1)
            self._write(bwd, 0)
        for pwm in [LF_PWM, LB_PWM, RF_PWM, RB_PWM]:
            self._set_pwm(pwm, PWM_FREQ, DUTY_SLOW)
        self._send_command("FORWARD_SLOW")

    def forward(self):
        for fwd, bwd in [(LF_FWD, LF_BWD), (LB_FWD, LB_BWD),
                         (RF_FWD, RF_BWD), (RB_FWD, RB_BWD)]:
            self._write(fwd, 1)
            self._write(bwd, 0)
        for pwm in [LF_PWM, LB_PWM, RF_PWM, RB_PWM]:
            self._set_pwm(pwm, PWM_FREQ, DUTY_STRAIGHT)
        self._send_command("FORWARD")

    def turn_left(self):
        for fwd, bwd in [(LF_FWD, LF_BWD), (LB_FWD, LB_BWD)]:
            self._write(fwd, 0)
            self._write(bwd, 0)
        for fwd, bwd in [(RF_FWD, RF_BWD), (RB_FWD, RB_BWD)]:
            self._write(fwd, 1)
            self._write(bwd, 0)
        self._set_pwm(LF_PWM, PWM_FREQ, DUTY_LEFT_STR)
        self._set_pwm(LB_PWM, PWM_FREQ, DUTY_LEFT_STR)
        self._set_pwm(RF_PWM, PWM_FREQ, DUTY_RIGHT_STR)
        self._set_pwm(RB_PWM, PWM_FREQ, DUTY_RIGHT_STR)
        self._send_command("LEFT")

    def turn_right(self):
        for fwd, bwd in [(LF_FWD, LF_BWD), (LB_FWD, LB_BWD)]:
            self._write(fwd, 1)
            self._write(bwd, 0)
        for fwd, bwd in [(RF_FWD, RF_BWD), (RB_FWD, RB_BWD)]:
            self._write(fwd, 0)
            self._write(bwd, 0)
        self._set_pwm(LF_PWM, PWM_FREQ, DUTY_RIGHT_STR)
        self._set_pwm(LB_PWM, PWM_FREQ, DUTY_RIGHT_STR)
        self._set_pwm(RF_PWM, PWM_FREQ, DUTY_LEFT_STR)
        self._set_pwm(RB_PWM, PWM_FREQ, DUTY_LEFT_STR)
        self._send_command("RIGHT")

    def cleanup(self):
        self.stop()
        if self.use_pigpio and self.pi:
            self.pi.stop()


def get_line_params(line):
    x1, y1, x2, y2 = line
    if x2 == x1:
        k = 1e6
    else:
        k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    return k, b


def filter_and_merge_lines(lines):
    valid = []
    kb_list = []
    for l in lines:
        k, b = get_line_params(l)
        if abs(k) < 0.1:
            continue
        ang = abs(math.atan(k) * 180 / math.pi)
        if ang < 20:
            continue
        dup = False
        for kb in kb_list:
            if abs(k - kb[0]) < 0.18 and abs(b - kb[1]) < 50:
                dup = True
                break
        if not dup:
            valid.append(l)
            kb_list.append((k, b))
    return valid


def draw_lanes_with_center_line(frame, lines):
    left_pts = []
    right_pts = []
    h, w = frame.shape[:2]
    y_low = h
    y_high = h // 2
    center_offset = 0
    single_line_offset = 0

    for l in lines:
        k, b = get_line_params(l)
        p1 = (l[0], l[1])
        p2 = (l[2], l[3])
        if k < 0:
            left_pts.append(p1)
            left_pts.append(p2)
        else:
            right_pts.append(p1)
            right_pts.append(p2)

    kl, bl, kr, br = 0, 0, 0, 0
    has_left = False
    has_right = False

    if len(left_pts) >= 2:
        pts = np.array(left_pts, dtype=np.int32).reshape(-1, 1, 2)
        lp = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        kl = lp[1][0] / lp[0][0]
        bl = lp[3][0] - kl * lp[2][0]
        has_left = True

    if len(right_pts) >= 2:
        pts = np.array(right_pts, dtype=np.int32).reshape(-1, 1, 2)
        rp = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        kr = rp[1][0] / rp[0][0]
        br = rp[3][0] - kr * rp[2][0]
        has_right = True

    if has_left:
        x1 = int((y_low - bl) / kl)
        x2 = int((y_high - bl) / kl)
        cv2.line(frame, (x1, y_low), (x2, y_high), (255, 255, 255), 2)

    if has_right:
        x1 = int((y_low - br) / kr)
        x2 = int((y_high - br) / kr)
        cv2.line(frame, (x1, y_low), (x2, y_high), (255, 255, 255), 2)

    if len(lines) == 1:
        k, b = get_line_params(lines[0])
        cx = w // 2
        x1 = cx
        y1 = y_low
        x2 = int(x1 - (y1 - y_high) / k)
        y2 = y_high
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        target_x = int((y_low - b) / k)
        single_line_offset = target_x - cx

    if has_left and has_right:
        cx1 = int(((y_low - bl) / kl + (y_low - br) / kr) / 2)
        cx2 = int(((y_high - bl) / kl + (y_high - br) / kr) / 2)
        cv2.line(frame, (cx1, y_low), (cx2, y_high), (0, 255, 0), 3)
        center_offset = cx1 - w // 2

    return center_offset, single_line_offset


def draw_objects(img, boxes):
    for box in boxes:
        cate_id = box.cate + 1
        if cate_id not in INTERESTED_CLASSES:
            continue

        text = f"{CLASS_NAMES[cate_id]} {box.score * 100:.1f}%"

        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x = box.x1
        y = box.y1 - th - baseline
        if y < 0:
            y = 0
        if x + tw > img.shape[1]:
            x = img.shape[1] - tw

        cv2.rectangle(img, (x, y), (x + tw, y + th + baseline), (255, 255, 255), -1)
        cv2.putText(img, text, (x, y + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv2.rectangle(img, (box.x1, box.y1), (box.x2, box.y2), (255, 0, 0))


def region_of_interest(img):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    pts = np.array([
        [0, h],
        [w - 1, h],
        [w - 1, h * 2 // 3],
        [0, h * 2 // 3]
    ], dtype=np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    return cv2.bitwise_and(img, mask)


def main():
    global last_offset

    motor = MotorControl(use_pigpio=True, control_url="http://172.20.10.5:5000/control")
    motor.stop()

    yolo = YoloFastestV2()
    if NCNN_AVAILABLE:
        yolo.load_model("yolo-fastestv2-opt.param", "yolo-fastestv2-opt.bin")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    if not cap.isOpened():
        print("摄像头打开失败！")
        return -1

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        gray = frame.copy()
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.Canny(gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
        gray = region_of_interest(gray)

        lines = cv2.HoughLinesP(gray, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
                                np.array([]), minLineLength=HOUGH_MIN_LINE_LENGTH,
                                maxLineGap=HOUGH_MAX_LINE_GAP)

        valid = []
        if lines is not None:
            lines_list = [l[0].tolist() for l in lines]
            valid = filter_and_merge_lines(lines_list)

        control_err = 0
        single_offset = 0

        if len(valid) == 0:
            motor.forward()
            print("无车道线 → 停车")
        else:
            dual_offset, single_offset = draw_lanes_with_center_line(frame, valid)

            if len(valid) == 1:
                control_err = single_offset
                print(f"沿蓝线行驶 | 偏移: {control_err}")
            else:
                control_err = dual_offset
                print(f"沿绿线行驶 | 偏移: {control_err}")

            smooth = 0.85
            smooth_err = int(control_err * (1 - smooth) + last_offset * smooth)
            last_offset = smooth_err

            dead = 25

            if len(valid) == 1:
                if abs(smooth_err) <= dead:
                    motor.forward_slow()
                elif smooth_err < -dead:
                    motor.turn_right()
                else:
                    motor.turn_left()
            else:
                if abs(smooth_err) <= dead:
                    motor.forward()
                elif smooth_err < -dead:
                    motor.turn_right()
                else:
                    motor.turn_left()

        cv2.imshow("edge", gray)
        cv2.imshow("lane", frame)
        if cv2.waitKey(10) == 27:
            motor.stop()
            break

    cap.release()
    cv2.destroyAllWindows()
    motor.cleanup()
    return 0


if __name__ == "__main__":
    main()
