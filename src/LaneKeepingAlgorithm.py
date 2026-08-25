import cv2
import numpy as np
import math
import sys
import time
# import RPi.GPIO as GPIO
# import torch
# import model.detector
# import utils.utils
import requests

stream_url = "http://172.20.10.5:8080/?action=stream"
control_url = "http://172.20.10.5:5000/control"

CAMERA_BRIGHTNESS_GAIN = 1.1

def get_hsv_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    hsv_mask = cv2.inRange(hsv, lower_white, upper_white)

    bgr_lower_white = np.array([180, 180, 180])
    bgr_upper_white = np.array([255, 255, 255])
    rgb_mask = cv2.inRange(frame, bgr_lower_white, bgr_upper_white)

    mask = cv2.bitwise_and(hsv_mask, rgb_mask)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return mask

def validate_curves_with_hsv(lane_curves, hsv_mask, search_radius=10, min_valid_ratio=0.3):
    left_curve, right_curve = lane_curves
    height, width = hsv_mask.shape

    def validate_single_curve(curve):
        if curve is None:
            return None

        valid_points = []
        for x, y in curve:
            if 0 <= y < height and 0 <= x < width:
                x_min = max(0, x - search_radius)
                x_max = min(width, x + search_radius + 1)
                y_min = max(0, y - search_radius)
                y_max = min(height, y + search_radius + 1)

                region = hsv_mask[y_min:y_max, x_min:x_max]
                white_ratio = np.count_nonzero(region) / region.size if region.size > 0 else 0

                if white_ratio >= min_valid_ratio:
                    valid_points.append((x, y))

        if len(valid_points) < 3:
            return None

        return valid_points

    validated_left = validate_single_curve(left_curve)
    validated_right = validate_single_curve(right_curve)

    return (validated_left, validated_right)

def detect_edges(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    hsv_mask = cv2.inRange(hsv, lower_white, upper_white)

    bgr_lower_white = np.array([180, 180, 180])
    bgr_upper_white = np.array([255, 255, 255])
    rgb_mask = cv2.inRange(frame, bgr_lower_white, bgr_upper_white)

    mask = cv2.bitwise_and(hsv_mask, rgb_mask)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    edges = cv2.Canny(mask, 50, 150)

    kernel_small = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_small)

    return edges

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height),
        (0, height * 0.3),
        (width, height * 0.3),
        (width, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)

    cropped_edges = cv2.bitwise_and(edges, mask)

    return cropped_edges

def detect_line_segments(cropped_edges):
    rho = 1
    theta = np.pi / 180
    min_threshold = 8

    line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold,
                                    np.array([]), minLineLength=2, maxLineGap=150)

    return line_segments

def thicken_lines(edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thickened = cv2.dilate(edges, kernel, iterations=2)
    thickened = cv2.morphologyEx(thickened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return thickened

def detect_lane_curves(edges):
    height, width = edges.shape
    roi_top = int(height * 0.35)

    roi_edges = edges[roi_top:, :]
    roi_height, roi_width = roi_edges.shape

    histogram = np.sum(roi_edges[roi_height // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    if histogram[left_base] < 30:
        left_base = int(width * 0.15)
    if histogram[right_base] < 30:
        right_base = int(width * 0.85)

    nwindows = 10
    window_height = roi_height // nwindows
    margin = 30
    min_pixels = 15

    leftx_current = left_base
    rightx_current = right_base

    left_lane_inds = []
    right_lane_inds = []

    nonzero = roi_edges.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeros = np.array(nonzero[0])

    for window in range(nwindows):
        win_y_low = roi_height - (window + 1) * window_height
        win_y_high = roi_height - window * window_height

        win_xleft_low = max(0, leftx_current - margin)
        win_xleft_high = min(roi_width, leftx_current + margin)
        win_xright_low = max(0, rightx_current - margin)
        win_xright_high = min(roi_width, rightx_current + margin)

        good_left_inds = ((nonzeros >= win_y_low) & (nonzeros < win_y_high) &
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeros >= win_y_low) & (nonzeros < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > min_pixels:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > min_pixels:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([])
    right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([])

    leftx = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
    lefty = nonzeros[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
    rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
    righty = nonzeros[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])

    left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 3 else None
    right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 3 else None

    left_curve = None
    right_curve = None

    if left_fit is not None:
        ploty = np.linspace(0, roi_height - 1, roi_height)
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        left_curve = []
        for i in range(0, len(ploty), 5):
            if 0 <= left_fitx[i] < roi_width:
                left_curve.append((int(left_fitx[i]), int(ploty[i]) + roi_top))

    if right_fit is not None:
        ploty = np.linspace(0, roi_height - 1, roi_height)
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        right_curve = []
        for i in range(0, len(ploty), 5):
            if 0 <= right_fitx[i] < roi_width:
                right_curve.append((int(right_fitx[i]), int(ploty[i]) + roi_top))

    return left_curve, right_curve

def detect_lane_lines_full(edges, frame):
    height, width = edges.shape
    rho = 1
    theta = np.pi / 180
    min_threshold = 15
    min_line_length = 40
    max_line_gap = 50

    lines = cv2.HoughLinesP(edges, rho, theta, min_threshold,
                           np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is None:
        return []

    lane_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if line_length < min_line_length:
                continue

            if x1 == x2:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if abs(slope) < 0.1:
                continue

            lane_lines.append([[x1, y1, x2, y2]])

    return lane_lines

def average_slope_intercept(frame, line_segments):
    lane_lines = []

    if line_segments is None:
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue

            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)

            if abs(slope) < 0.15:
                continue

            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        lane_lines.append(make_points(frame, left_fit_average))

    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        lane_lines.append(make_points(frame, right_fit_average))

    return lane_lines

def make_points(frame, line):
    height, width, _ = frame.shape
    
    slope, intercept = line
    
    y1 = height  # bottom of the frame
    y2 = int(y1 / 2)  # make points from middle of the frame down
    
    if slope == 0:
        slope = 0.1
        
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return [[x1, y1, x2, y2]]

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=6):
    line_image = np.zeros_like(frame)

    if lines is None or len(lines) == 0:
        return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    if isinstance(lines, tuple) and len(lines) == 2:
        left_curve, right_curve = lines
        if left_curve:
            for i in range(len(left_curve) - 1):
                x1, y1 = left_curve[i]
                x2, y2 = left_curve[i + 1]
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
        if right_curve:
            for i in range(len(right_curve) - 1):
                x1, y1 = right_curve[i]
                x2, y2 = right_curve[i + 1]
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    else:
        for line in lines:
            if line is None or len(line) == 0:
                continue
            for x1, y1, x2, y2 in line:
                if not all(isinstance(v, (int, float)) for v in [x1, y1, x2, y2]):
                    continue
                cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), line_color, line_width)

    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    return line_image

def display_heading_line(frame, lane_input, steering_angle, line_color=(0, 0, 255), line_width=5):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    if isinstance(lane_input, tuple) and len(lane_input) == 2:
        left_curve, right_curve = lane_input
        left_pts = left_curve if left_curve else []
        right_pts = right_curve if right_curve else []

        if left_pts and right_pts:
            combined = list(left_pts) + list(right_pts)
            combined.sort(key=lambda p: p[1], reverse=True)

            prev_x = None
            for i in range(len(combined) - 1):
                x1, y1 = combined[i]
                x2, y2 = combined[i + 1]
                if abs(y1 - y2) < 30 and abs(x1 - x2) < 100:
                    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)

            heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    elif lane_input is not None and len(lane_input) >= 2:
        left_line = lane_input[0][0]
        right_line = lane_input[1][0]

        prev_x = None
        for y in range(height, int(height * 0.3), -5):
            if len(left_line) == 6 and len(right_line) == 6:
                a_l, b_l, c_l = left_line[4], left_line[5], left_line[0] - left_line[4] * left_line[1]**2 - left_line[5] * left_line[1]
                a_r, b_r, c_r = right_line[4], right_line[5], right_line[0] - right_line[4] * right_line[1]**2 - right_line[5] * right_line[1]
                x_left = int(a_l * y**2 + b_l * y + c_l)
                x_right = int(a_r * y**2 + b_r * y + c_r)
            else:
                x1_l, y1_l, x2_l, y2_l = left_line[:4]
                x1_r, y1_r, x2_r, y2_r = right_line[:4]

                slope_l = (y2_l - y1_l) / (x2_l - x1_l) if x2_l != x1_l else 0.001
                slope_r = (y2_r - y1_r) / (x2_r - x1_r) if x2_r != x1_r else 0.001

                intercept_l = y1_l - slope_l * x1_l
                intercept_r = y1_r - slope_r * x1_r

                x_left = int((y - intercept_l) / slope_l)
                x_right = int((y - intercept_r) / slope_r)

            x_center = (x_left + x_right) // 2

            if prev_x is not None and abs(x_center - prev_x) < 50:
                cv2.line(heading_image, (prev_x, y + 5), (x_center, y), line_color, line_width)

            prev_x = x_center

        heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    elif lane_input is not None and len(lane_input) == 1:
        heading_image = frame.copy()
        line = lane_input[0][0]
        
        if len(line) == 6:
            a, b = line[4], line[5]
            c = line[0] - a * line[1]**2 - b * line[1]
            
            prev_x = None
            for y in range(height, int(height * 0.3), -5):
                x = int(a * y**2 + b * y + c)
                if prev_x is not None:
                    cv2.line(heading_image, (prev_x, y + 5), (x, y), line_color, line_width)
                prev_x = x
        else:
            x1, y1, x2, y2 = line[:4]
            cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    else:
        steering_angle_radian = steering_angle / 180.0 * math.pi
        x1 = int(width / 2)
        y1 = height
        tan_val = math.tan(steering_angle_radian)
        if abs(tan_val) < 0.001:
            tan_val = 0.001
        x2 = int(x1 - height / 2 / tan_val)
        y2 = int(height / 2)
        cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
        heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image

def combine_windows_2x2(img1, img2, img3, img4, labels=None):
    h, w = img1.shape[:2]

    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    if len(img3.shape) == 2:
        img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)
    if len(img4.shape) == 2:
        img4 = cv2.cvtColor(img4, cv2.COLOR_GRAY2BGR)

    top = np.hstack([img1, img2])
    bottom = np.hstack([img3, img4])
    combined = np.vstack([top, bottom])

    if labels:
        for i, (label, (x, y)) in enumerate(labels.items()):
            cv2.putText(combined, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return combined

def get_steering_angle(frame, lane_input):

    height, width, _ = frame.shape

    if isinstance(lane_input, tuple) and len(lane_input) == 2:
        left_curve, right_curve = lane_input
        if left_curve and right_curve:
            left_pts = [(p[0], p[1]) for p in left_curve]
            right_pts = [(p[0], p[1]) for p in right_curve]
            left_bottom = max(left_pts, key=lambda p: p[1])
            right_bottom = max(right_pts, key=lambda p: p[1])
            lane_center_x = (left_bottom[0] + right_bottom[0]) / 2
        elif left_curve:
            left_pts = [(p[0], p[1]) for p in left_curve]
            left_bottom = max(left_pts, key=lambda p: p[1])
            lane_center_x = left_bottom[0] + width * 0.2
        elif right_curve:
            right_pts = [(p[0], p[1]) for p in right_curve]
            right_bottom = max(right_pts, key=lambda p: p[1])
            lane_center_x = right_bottom[0] - width * 0.2
        else:
            lane_center_x = width / 2
        x_offset = lane_center_x - width / 2
        y_offset = height / 2

    elif isinstance(lane_input, list) and len(lane_input) >= 2:
        left_line = lane_input[0]
        right_line = lane_input[1]
        if len(left_line) > 0 and len(right_line) > 0:
            left_x2 = left_line[0][2]
            right_x2 = right_line[0][2]
            lane_center_x = (left_x2 + right_x2) / 2
        elif len(left_line) > 0:
            lane_center_x = left_line[0][2] + width * 0.2
        elif len(right_line) > 0:
            lane_center_x = right_line[0][2] - width * 0.2
        else:
            lane_center_x = width / 2
        x_offset = lane_center_x - width / 2
        y_offset = height / 2

    elif isinstance(lane_input, list) and len(lane_input) == 1:
        x1, y1, x2, y2 = lane_input[0][0]
        x_offset = x2 - x1
        y_offset = height / 2

    else:
        x_offset = 0
        y_offset = height / 2

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90

    return steering_angle

lane_history = []
LANE_HISTORY_SIZE = 5

frame_buffer = []
FRAME_BUFFER_SIZE = 3

steering_history = []
STEERING_HISTORY_SIZE = 3

last_valid_command = "FORWARD"

video = cv2.VideoCapture(stream_url)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
video.set(cv2.CAP_PROP_FPS, 30)

video.set(cv2.CAP_PROP_EXPOSURE, 50)
video.set(cv2.CAP_PROP_GAIN, 1.0)

time.sleep(1)

while True:
    ret, frame = video.read()
    if not ret or frame is None or frame.size == 0:
        continue

    if CAMERA_BRIGHTNESS_GAIN != 1.0:
        frame = np.clip(frame * CAMERA_BRIGHTNESS_GAIN, 0, 255).astype(np.uint8)

    frame_buffer.append(frame.astype(np.float32))
    if len(frame_buffer) > FRAME_BUFFER_SIZE:
        frame_buffer.pop(0)

    if len(frame_buffer) == FRAME_BUFFER_SIZE:
        frame = np.mean(frame_buffer, axis=0).astype(np.uint8)

    cv2.imshow("original", frame)
    hsv_mask = get_hsv_mask(frame)
    edges = detect_edges(frame)
    solid_edges = thicken_lines(edges)
    roi = region_of_interest(solid_edges)
    lane_curves = detect_lane_curves(solid_edges)
    lane_curves = validate_curves_with_hsv(lane_curves, hsv_mask)

    lane_history.append(lane_curves)
    if len(lane_history) > LANE_HISTORY_SIZE:
        lane_history.pop(0)

    if len(lane_history) >= 3:
        valid_count = sum(1 for lc in lane_history if lc[0] is not None or lc[1] is not None)
        if valid_count >= 2:
            left_curves = [lc[0] for lc in lane_history if lc[0] is not None]
            right_curves = [lc[1] for lc in lane_history if lc[1] is not None]
            avg_left = left_curves[len(left_curves) // 2] if left_curves else None
            avg_right = right_curves[len(right_curves) // 2] if right_curves else None
            if avg_left or avg_right:
                lane_curves = (avg_left, avg_right)

    lane_lines_image = display_lines(frame, lane_curves)
    steering_angle = get_steering_angle(frame, lane_curves)

    steering_history.append(steering_angle)
    if len(steering_history) > STEERING_HISTORY_SIZE:
        steering_history.pop(0)

    smoothed_steering = int(np.mean(steering_history))

    heading_image = display_heading_line(lane_lines_image, lane_curves, smoothed_steering)

    edges_with_lanes = display_lines(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), lane_curves)
    roi_with_lanes = display_lines(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), lane_curves)

    combined = combine_windows_2x2(frame, edges_with_lanes, roi_with_lanes, heading_image)
    cv2.imshow("Lane Keeping (2x2)", combined)

    deviation = smoothed_steering - 90

    if abs(deviation) > 45:
        print(f"[KEEP] red_line_angle={smoothed_steering}, deviation={deviation}, keep={last_valid_command}")
        try:
            response = requests.post(control_url, json={'command': last_valid_command}, timeout=1)
        except Exception as e:
            print(f"Connection error: {e}")

    elif abs(deviation) < 5:
        last_valid_command = "FORWARD"
        print(f"[FORWARD] red_line_angle={smoothed_steering}, deviation={deviation}")
        try:
            response = requests.post(control_url, json={'command': "FORWARD"}, timeout=1)
        except Exception as e:
            print(f"Connection error: {e}")

    elif deviation > 5:
        last_valid_command = "RIGHT"
        print(f"[RIGHT] red_line_angle={smoothed_steering}, deviation={deviation}")
        try:
            response = requests.post(control_url, json={'command': "RIGHT"}, timeout=1)
        except Exception as e:
            print(f"Connection error: {e}")

    elif deviation < -5:
        last_valid_command = "LEFT"
        print(f"[LEFT] red_line_angle={smoothed_steering}, deviation={deviation}")
        try:
            response = requests.post(control_url, json={'command': "LEFT"}, timeout=1)
        except Exception as e:
            print(f"Connection error: {e}")

    key = cv2.waitKey(1)
    if key == 27:
        break
    
video.release()
cv2.destroyAllWindows()

