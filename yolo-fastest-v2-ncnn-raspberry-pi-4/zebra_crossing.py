import cv2
import numpy as np
import signal
import time
from collections import deque
import requests

import ncnn

CONTROL_URL = "http://172.20.10.5:5000/control"
STREAM_URL = "http://172.20.10.5:8080/?action=stream"

BASE_SPEED = 200000
SLOW_SPEED = 140000

DEVIATION_THRESHOLD = 20
CHECK_COUNT = 5
CHECK_TIME = 0.5

ZEBRA_WHITE_RATIO_THRESHOLD = 0.35
ZEBRA_HORIZONTAL_SPREAD = 0.6
ZEBRA_STOP_DURATION = 3.0
ZEBRA_COOLDOWN = 5.0

W, H = 320, 240


class TargetBox:
    def __init__(self, x1=0, y1=0, x2=0, y2=0, cate=-1, score=0.0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.cate = cate
        self.score = score

    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class YoloFastestV2:
    def __init__(self):
        self.num_output = 2
        self.num_threads = 4
        self.num_anchor = 3
        self.num_category = 80
        self.nms_thresh = 0.25
        self.input_width = 352
        self.input_height = 352
        self.anchor = [
            12.64, 19.39, 37.88, 51.48, 55.71, 138.31,
            126.91, 78.23, 131.57, 214.55, 279.92, 258.87
        ]
        self.net = ncnn.Net()

    def init(self, use_vulkan=False):
        self.net.opt.use_winograd_convolution = True
        self.net.opt.use_sgemm_convolution = True
        self.net.opt.use_int8_inference = True
        self.net.opt.use_vulkan_compute = use_vulkan
        self.net.opt.num_threads = self.num_threads
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_storage = True
        self.net.opt.use_fp16_arithmetic = True
        self.net.opt.use_int8_storage = True
        self.net.opt.use_int8_arithmetic = True
        self.net.opt.use_packing_layout = True

    def load_model(self, param_path, bin_path):
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        print("Ncnn model init success...")

    @staticmethod
    def _intersection_area(a, b):
        if a.x1 > b.x2 or a.x2 < b.x1 or a.y1 > b.y2 or a.y2 < b.y1:
            return 0.0
        inter_w = min(a.x2, b.x2) - max(a.x1, b.x1)
        inter_h = min(a.y2, b.y2) - max(a.y1, b.y1)
        return inter_w * inter_h

    def _nms(self, tmp_boxes):
        tmp_boxes.sort(key=lambda b: b.score, reverse=True)
        picked = []
        for i in range(len(tmp_boxes)):
            keep = True
            for j in picked:
                inter = self._intersection_area(tmp_boxes[i], tmp_boxes[j])
                union = tmp_boxes[i].area() + tmp_boxes[j].area() - inter
                iou = inter / union if union > 0 else 0
                if iou > self.nms_thresh and tmp_boxes[i].cate == tmp_boxes[j].cate:
                    keep = False
                    break
            if keep:
                picked.append(i)
        return [tmp_boxes[i] for i in picked]

    def _get_category(self, values, index):
        tmp = 0.0
        obj_score = values[4 * self.num_anchor + index]
        category = -1
        score = -1.0
        for i in range(self.num_category):
            cls_score = values[4 * self.num_anchor + self.num_anchor + i]
            cls_score *= obj_score
            if cls_score > tmp:
                score = cls_score
                category = i
                tmp = cls_score
        return category, score

    def _pred_handle(self, outs, scale_w, scale_h, thresh):
        dst_boxes = []
        for i in range(self.num_output):
            out = outs[i]
            out_h = out.c
            out_w = out.h
            out_c = out.w
            stride = self.input_height // out_h

            for h_idx in range(out_h):
                values = np.array(out.channel(h_idx)).reshape(-1)
                for w_idx in range(out_w):
                    offset = w_idx * out_c
                    for b in range(self.num_anchor):
                        category, score = self._get_category(values[offset:], b)
                        if score > thresh:
                            bcx = ((values[offset + b * 4 + 0] * 2.0 - 0.5) + w_idx) * stride
                            bcy = ((values[offset + b * 4 + 1] * 2.0 - 0.5) + h_idx) * stride
                            bw = (values[offset + b * 4 + 2] * 2.0) ** 2 * self.anchor[i * self.num_anchor * 2 + b * 2 + 0]
                            bh = (values[offset + b * 4 + 3] * 2.0) ** 2 * self.anchor[i * self.num_anchor * 2 + b * 2 + 1]

                            box = TargetBox(
                                x1=(bcx - 0.5 * bw) * scale_w,
                                y1=(bcy - 0.5 * bh) * scale_h,
                                x2=(bcx + 0.5 * bw) * scale_w,
                                y2=(bcy + 0.5 * bh) * scale_h,
                                cate=category,
                                score=score
                            )
                            dst_boxes.append(box)
        return dst_boxes

    def detection(self, src_img, thresh=0.3):
        scale_w = src_img.shape[1] / self.input_width
        scale_h = src_img.shape[0] / self.input_height

        mat_in = ncnn.Mat.from_pixels_resize(
            src_img, ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            src_img.shape[1], src_img.shape[0],
            self.input_width, self.input_height
        )

        mean_vals = [0.0, 0.0, 0.0]
        norm_vals = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
        mat_in.substract_mean_normalize(mean_vals, norm_vals)

        ex = self.net.create_extractor()
        ex.input("input.1", mat_in)

        out0, ret0 = ex.extract("794")
        out1, ret1 = ex.extract("796")

        tmp_boxes = self._pred_handle([out0, out1], scale_w, scale_h, thresh)
        dst_boxes = self._nms(tmp_boxes)
        return dst_boxes


def get_white_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

    lower_white_hsv = np.array([0, 0, 180])
    upper_white_hsv = np.array([180, 40, 255])
    hsv_mask = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)

    lower_white_bgr = np.array([180, 180, 180])
    upper_white_bgr = np.array([255, 255, 255])
    bgr_mask = cv2.inRange(frame, lower_white_bgr, upper_white_bgr)

    mask = cv2.bitwise_and(hsv_mask, bgr_mask)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return mask


def detect_zebra_crossing(mask):
    roi_top = int(H * 0.5)
    roi_bottom = int(H * 0.85)
    roi = mask[roi_top:roi_bottom, :]

    roi_area = roi.shape[0] * roi.shape[1]
    white_pixels = cv2.countNonZero(roi)
    white_ratio = white_pixels / roi_area

    if white_pixels > 0:
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_points = np.vstack(contours)
            x_min = all_points[:, 0, 0].min()
            x_max = all_points[:, 0, 0].max()
            horizontal_spread = (x_max - x_min) / W
        else:
            horizontal_spread = 0.0
    else:
        horizontal_spread = 0.0

    is_zebra = white_ratio > ZEBRA_WHITE_RATIO_THRESHOLD and horizontal_spread > ZEBRA_HORIZONTAL_SPREAD

    return is_zebra, white_ratio, horizontal_spread


class CarController:
    def __init__(self, control_url):
        self.control_url = control_url
        self.need_exit = False
        self.last_valid_command = "STOP"

        self.err_history = deque(maxlen=8)
        self.last_check_times = deque(maxlen=CHECK_COUNT)
        self.current_speed = BASE_SPEED
        self.is_corner = False

        self.zebra_stopping = False
        self.zebra_stop_start = 0.0
        self.zebra_last_detected = 0.0

        signal.signal(signal.SIGINT, self._handle_stop)

    def send_command(self, command, left=None, right=None):
        payload = {'command': command}
        if left is not None:
            payload['left'] = left
        if right is not None:
            payload['right'] = right
        try:
            response = requests.post(self.control_url, json=payload, timeout=1)
            self.last_valid_command = command
            print(f"[{command}] left={left} right={right}, status={response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[{command}] connection error: {e}")

    def car_stop(self):
        self.send_command("STOP")

    def _handle_stop(self, sig, frame):
        self.need_exit = True
        self.car_stop()

    def smooth_error(self, raw_err):
        self.err_history.append(raw_err)
        return int(sum(self.err_history) / len(self.err_history))

    def check_corner(self, err):
        now_sec = time.monotonic()
        self.last_check_times.append(now_sec)

        if not hasattr(self, '_dev_cnt'):
            self._dev_cnt = 0

        if abs(err) > DEVIATION_THRESHOLD:
            self._dev_cnt += 1
        else:
            self._dev_cnt = 0

        time_ok = False
        if len(self.last_check_times) == CHECK_COUNT:
            dt = self.last_check_times[-1] - self.last_check_times[0]
            time_ok = dt <= CHECK_TIME

        if self._dev_cnt >= CHECK_COUNT and time_ok:
            self.is_corner = True
            self.current_speed = SLOW_SPEED
        elif self._dev_cnt == 0:
            self.is_corner = False
            self.current_speed = BASE_SPEED

    def pure_pursuit(self, center_err):
        max_err = 100
        center_err = max(-max_err, min(center_err, max_err))

        k = 0.7
        left = self.current_speed - center_err * k
        right = self.current_speed + center_err * k

        left = max(120000, min(int(left), 250000))
        right = max(120000, min(int(right), 250000))

        self.send_command("DRIVE", left=left, right=right)

    def handle_zebra(self, is_zebra):
        now = time.monotonic()

        if self.zebra_stopping:
            elapsed = now - self.zebra_stop_start
            if elapsed >= ZEBRA_STOP_DURATION:
                self.zebra_stopping = False
                self.zebra_last_detected = now
                print("[ZEBRA] stop finished, resuming")
            else:
                remaining = ZEBRA_STOP_DURATION - elapsed
                print(f"[ZEBRA] stopping... {remaining:.1f}s remaining")
            return True

        if is_zebra and (now - self.zebra_last_detected) > ZEBRA_COOLDOWN:
            self.zebra_stopping = True
            self.zebra_stop_start = now
            self.car_stop()
            print("[ZEBRA] detected! stopping for 3 seconds")
            return True

        return False

    def cleanup(self):
        self.car_stop()


def get_lane_points(binary):
    pts = []
    for y in range(H - 80, H):
        for x in range(W):
            if binary[y, x] > 128:
                pts.append((x, y))
    return pts


def fit_poly(pts, order=2):
    if len(pts) < 5:
        return np.zeros(3)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    A = np.vstack([ys ** 2, ys, np.ones(len(ys))]).T
    coeff, _, _, _ = np.linalg.lstsq(A, xs, rcond=None)
    return coeff


def calc_x(coeff, y):
    a, b, c = coeff[0], coeff[1], coeff[2]
    return int(a * y * y + b * y + c)


def main():
    car = CarController(CONTROL_URL)
    car.car_stop()

    yolo = YoloFastestV2()
    yolo.init()
    yolo.load_model("yolo-fastestv2-opt.param", "yolo-fastestv2-opt.bin")

    cap = cv2.VideoCapture(STREAM_URL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    while not car.need_exit:
        ret, frame = cap.read()
        if not ret:
            continue

        white_mask = get_white_mask(frame)
        is_zebra, white_ratio, h_spread = detect_zebra_crossing(white_mask)
        zebra_handling = car.handle_zebra(is_zebra)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 100)

        pts = get_lane_points(canny)
        coeff = fit_poly(pts)

        target_y = H - 20
        cx = calc_x(coeff, target_y)
        center = W // 2
        raw_err = cx - center

        smooth_err = car.smooth_error(raw_err)
        car.check_corner(raw_err)

        if not zebra_handling:
            if len(pts) > 20:
                car.pure_pursuit(smooth_err)
            else:
                car.car_stop()

        roi_top = int(H * 0.5)
        roi_bottom = int(H * 0.85)
        zebra_roi = white_mask[roi_top:roi_bottom, :]
        zebra_display = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(zebra_display, (0, roi_top), (W, roi_bottom), (255, 255, 0), 1)

        cv2.circle(frame, (cx, target_y), 6, (0, 255, 0), -1)
        cv2.line(frame, (center, H), (center, H - 60), (0, 0, 255), 2)

        cv2.putText(frame, f"Err: {smooth_err}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        if car.zebra_stopping:
            elapsed = time.monotonic() - car.zebra_stop_start
            remaining = ZEBRA_STOP_DURATION - elapsed
            cv2.putText(frame, f"ZEBRA! STOP {remaining:.1f}s",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif is_zebra:
            cv2.putText(frame, "ZEBRA DETECTED",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        elif car.is_corner:
            cv2.putText(frame, "CORNER - SLOW",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "NORMAL - RUN",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.putText(frame, f"White: {white_ratio:.2f} Spread: {h_spread:.2f}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        cv2.imshow("white_mask", zebra_display)
        cv2.imshow("edge", canny)
        cv2.imshow("lane", frame)

        if cv2.waitKey(10) == 27:
            car.car_stop()
            break

    car.cleanup()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
