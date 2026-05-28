import cv2
import numpy as np
import requests

from constants import STREAM_URL, CONTROL_URL

# Open camera — use 0 for USB webcam, or picamera2 for CSI
cap = cv2.VideoCapture(STREAM_URL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Camera not found")
    exit()

print("Camera opened successfully")

"""
For CSI cameras using picamera2 (recommended on Pi 5):

from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

frame = picam2.capture_array()
"""


def detect_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Lower threshold = 50, upper = 150 (adjust for your lighting)
    edges = cv2.Canny(blurred, 50, 150)

    return edges

def region_of_interest(img):
    height, width = img.shape
    # Triangle covering the lower 60% of the frame
    polygon = np.array([[
        (0, height),
        (width, height),
        (width // 2, int(height * 0.4))
    ]], np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def detect_lane_lines(masked_edges):
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,           # Distance resolution in pixels
        theta=np.pi/180, # Angle resolution in radians
        threshold=50,    # Minimum intersections to detect a line
        minLineLength=40,# Minimum line length in pixels
        maxLineGap=20    # Maximum gap between segments
    )
    return lines

def average_slope_intercept(frame, lines):
    """Average multiple detected segments into two lane lines."""
    left_fit = []
    right_fit = []
    if lines is None:
        return None, None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue  # Skip vertical lines
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        if abs(slope) >= 0.5:  # Right lane (positive slope)
            right_fit.append((slope, intercept))
    left_line = np.average(left_fit, axis=0) if left_fit else None
    right_line = np.average(right_fit, axis=0) if right_fit else None
    return left_line, right_line

def make_coordinates(frame, line_params):
    """Convert slope-intercept form to pixel coordinates."""
    if line_params is None:
        return None
    slope, intercept = line_params
    height = frame.shape[0]
    y1 = height              # Bottom of frame
    y2 = int(height * 0.6)  # Extend to 60% up
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

def draw_lines(frame, lines):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

def compute_steering(frame, left_line, right_line):
    """Returns steering angle in degrees. 0 = straight ahead."""
    height, width = frame.shape[:2]
    mid_x = width // 2

    if left_line is not None and right_line is not None:
        # Both lanes visible — average their x positions at mid-frame
        left_x = (left_line[0] + left_line[2]) // 2
        right_x = (right_line[0] + right_line[2]) // 2
        lane_centre = (left_x + right_x) // 2
    elif left_line is not None:
        lane_centre = left_line[0] + 200  # Estimated
    elif right_line is not None:
        lane_centre = right_line[0] - 200
    else:
        return 0  # No lanes detected, go straight

    error = lane_centre - mid_x
    # Simple proportional steering
    steering_angle = int(error * 0.1)  # Tune the gain
    return steering_angle

while True:
    ret, frame = cap.read()
    edges = detect_edges(frame)
    masked_edges = region_of_interest(edges)
    lines = detect_lane_lines(masked_edges)
    left_line, right_line = average_slope_intercept(frame, lines)
    cv2.imshow("processed lines", draw_lines(frame, lines))
    steering_angle = compute_steering(frame, left_line, right_line)

    if steering_angle > 0:
        response = requests.post(CONTROL_URL, json={'command': "RIGHT"})
        print("Turn right")
    elif steering_angle < 0:
        response = requests.post(CONTROL_URL, json={'command': "LEFT"})
        print("Turn left")
    else:
        response = requests.post(CONTROL_URL, json={'command': "FORWARD"})
        print("Forward")

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


