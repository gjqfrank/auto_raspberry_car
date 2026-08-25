#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <pigpiod_if2.h>
#include "yolo-fastestv2.h"

// ==================== 电机引脚定义 ====================
#define LF_FWD      1
#define LF_BWD      7
#define LF_PWM      12
#define LB_FWD      24
#define LB_BWD      23
#define LB_PWM      18
#define RF_FWD      6
#define RF_BWD      5
#define RF_PWM      13
#define RB_FWD      21
#define RB_BWD      20
#define RB_PWM      19

// ==================== PWM 参数 ====================
const int PWM_FREQ = 100;
const int DUTY_STRAIGHT = 130000;
const int DUTY_SLOW = 90000;
const int DUTY_LEFT_STR = 220000;
const int DUTY_RIGHT_STR = 90000;

// ==================== 全局变量 ====================
int pi;
int last_offset = 0;
yoloFastestv2 yoloF2;

// ==================== 【最高优先级】行人检测延时停车 ====================
const bool ENABLE_PEDESTRIAN_STOP = true;
const float PEDESTRIAN_STOP_DELAY_SEC = 1.0f;  // 可修改延时
bool person_detected = false;
std::chrono::steady_clock::time_point person_detect_time;

// ==================== 红绿矩形结构体 ====================
struct RectInfo {
    std::vector<cv::Point> contour;
    bool is_green;
};

// ==================== 状态机 ====================
enum CarState {
    STATE_STOP,
    STATE_GO
};

// ==================== 车道线参数 ====================
const unsigned int IMG_WIDTH = 320;
const unsigned int IMG_HEIGHT = 240;
const cv::Point mask_left_bottom(0, IMG_HEIGHT);
const cv::Point mask_right_bottom(IMG_WIDTH - 1, IMG_HEIGHT);
const cv::Point mask_left_middle(0, IMG_HEIGHT * 2 / 3);
const cv::Point mask_right_middle(IMG_WIDTH - 1, IMG_HEIGHT * 2 / 3);

const unsigned char canny_low_threshold = 60;
const unsigned char canny_high_threshold = 120;
const unsigned char hough_rho = 1;
const double hough_theta = CV_PI / 180;
const unsigned char hough_threshold = 35;
const unsigned char hough_min_line_length = 20;
const unsigned char hough_max_line_gap = 40;

// ==================== 类别ID ====================
const unsigned char PERSON_CLASS_ID = 1;

const char* class_names[] = {
    "background", "person", "bicycle", "car", "motorbike",
    "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign"
};

// ==================== 电机控制 ====================
void motor_init() {
    pi = pigpio_start(NULL, NULL);
    if (pi < 0) {
        std::cerr << "pigpio 连接失败！" << std::endl;
        exit(1);
    }
    set_mode(pi, LF_FWD, PI_OUTPUT); set_mode(pi, LF_BWD, PI_OUTPUT);
    set_mode(pi, LB_FWD, PI_OUTPUT); set_mode(pi, LB_BWD, PI_OUTPUT);
    set_mode(pi, RF_FWD, PI_OUTPUT); set_mode(pi, RF_BWD, PI_OUTPUT);
    set_mode(pi, RB_FWD, PI_OUTPUT); set_mode(pi, RB_BWD, PI_OUTPUT);
    set_mode(pi, LF_PWM, PI_OUTPUT); set_mode(pi, LB_PWM, PI_OUTPUT);
    set_mode(pi, RF_PWM, PI_OUTPUT); set_mode(pi, RB_PWM, PI_OUTPUT);
}

void car_stop() {
    gpio_write(pi, LF_FWD, 0); gpio_write(pi, LF_BWD, 0);
    gpio_write(pi, LB_FWD, 0); gpio_write(pi, LB_BWD, 0);
    gpio_write(pi, RF_FWD, 0); gpio_write(pi, RF_BWD, 0);
    gpio_write(pi, RB_FWD, 0); gpio_write(pi, RB_BWD, 0);
    hardware_PWM(pi, LF_PWM, 0, 0);
    hardware_PWM(pi, LB_PWM, 0, 0);
    hardware_PWM(pi, RF_PWM, 0, 0);
    hardware_PWM(pi, RB_PWM, 0, 0);
}

void car_forward_slow() {
    gpio_write(pi, LF_FWD, 1); gpio_write(pi, LF_BWD, 0);
    gpio_write(pi, LB_FWD, 1); gpio_write(pi, LB_BWD, 0);
    gpio_write(pi, RF_FWD, 1); gpio_write(pi, RF_BWD, 0);
    gpio_write(pi, RB_FWD, 1); gpio_write(pi, RB_BWD, 0);
    hardware_PWM(pi, LF_PWM, PWM_FREQ, DUTY_SLOW);
    hardware_PWM(pi, LB_PWM, PWM_FREQ, DUTY_SLOW);
    hardware_PWM(pi, RF_PWM, PWM_FREQ, DUTY_SLOW);
    hardware_PWM(pi, RB_PWM, PWM_FREQ, DUTY_SLOW);
}

void car_forward() {
    gpio_write(pi, LF_FWD, 1); gpio_write(pi, LF_BWD, 0);
    gpio_write(pi, LB_FWD, 1); gpio_write(pi, LB_BWD, 0);
    gpio_write(pi, RF_FWD, 1); gpio_write(pi, RF_BWD, 0);
    gpio_write(pi, RB_FWD, 1); gpio_write(pi, RB_BWD, 0);
    hardware_PWM(pi, LF_PWM, PWM_FREQ, DUTY_STRAIGHT);
    hardware_PWM(pi, LB_PWM, PWM_FREQ, DUTY_STRAIGHT);
    hardware_PWM(pi, RF_PWM, PWM_FREQ, DUTY_STRAIGHT);
    hardware_PWM(pi, RB_PWM, PWM_FREQ, DUTY_STRAIGHT);
}

void car_turn_left() {
    gpio_write(pi, LF_FWD, 1); gpio_write(pi, LF_BWD, 0);
    gpio_write(pi, LB_FWD, 1); gpio_write(pi, LB_BWD, 0);
    gpio_write(pi, RF_FWD, 1); gpio_write(pi, RF_BWD, 0);
    gpio_write(pi, RB_FWD, 1); gpio_write(pi, RB_BWD, 0);
    hardware_PWM(pi, LF_PWM, PWM_FREQ, DUTY_LEFT_STR);
    hardware_PWM(pi, LB_PWM, PWM_FREQ, DUTY_LEFT_STR);
    hardware_PWM(pi, RF_PWM, PWM_FREQ, DUTY_RIGHT_STR);
    hardware_PWM(pi, RB_PWM, PWM_FREQ, DUTY_RIGHT_STR);
}

void car_turn_right() {
    gpio_write(pi, LF_FWD, 1); gpio_write(pi, LF_BWD, 0);
    gpio_write(pi, LB_FWD, 1); gpio_write(pi, LB_BWD, 0);
    gpio_write(pi, RF_FWD, 1); gpio_write(pi, RF_BWD, 0);
    gpio_write(pi, RB_FWD, 1); gpio_write(pi, RB_BWD, 0);
    hardware_PWM(pi, LF_PWM, PWM_FREQ, DUTY_RIGHT_STR);
    hardware_PWM(pi, LB_PWM, PWM_FREQ, DUTY_RIGHT_STR);
    hardware_PWM(pi, RF_PWM, PWM_FREQ, DUTY_LEFT_STR);
    hardware_PWM(pi, RB_PWM, PWM_FREQ, DUTY_LEFT_STR);
}

// ==================== 红绿矩形识别 ====================
std::vector<RectInfo> detect_color_rect(cv::Mat& frame) {
    cv::Mat hsv;
    cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::Mat red1, red2, red_mask, green_mask;

    inRange(hsv, cv::Scalar(0, 150, 100), cv::Scalar(10, 255, 255), red1);
    inRange(hsv, cv::Scalar(170, 150, 100), cv::Scalar(180, 255, 255), red2);
    red_mask = red1 | red2;
    inRange(hsv, cv::Scalar(50, 100, 100), cv::Scalar(75, 255, 255), green_mask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    morphologyEx(red_mask, red_mask, cv::MORPH_CLOSE, kernel);
    morphologyEx(green_mask, green_mask, cv::MORPH_CLOSE, kernel);

    std::vector<RectInfo> res;
    std::vector<std::vector<cv::Point>> contours;

    auto find = [&](cv::Mat& m, bool is_green) {
        contours.clear();
        cv::findContours(m, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (auto& c : contours) {
            if (contourArea(c) < 500) continue;
            std::vector<cv::Point> approx;
            approxPolyDP(c, approx, 0.04 * arcLength(c, true), true);
            if (approx.size() == 4) res.push_back({ approx, is_green });
        }
    };
    find(red_mask, false);
    find(green_mask, true);
    return res;
}

// ==================== 车道线工具 ====================
void getLineParams(cv::Vec4i line, double &k, double &b) {
    int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
    k = (x2 == x1) ? 1e6 : (double)(y2 - y1) / (x2 - x1);
    b = y1 - k * x1;
}

std::vector<cv::Vec4i> filterAndMergeLines(std::vector<cv::Vec4i> &lines) {
    std::vector<cv::Vec4i> valid;
    for (auto& l : lines) {
        double k, b; getLineParams(l, k, b);
        if (fabs(k) < 0.3) continue;
        if (fabs(atan(k) * 180 / CV_PI) < 25) continue;
        valid.push_back(l);
    }
    return valid;
}

int drawLanesWithCenterLine(cv::Mat &frame, std::vector<cv::Vec4i> &lines, int &single_line_offset) {
    std::vector<cv::Point> l_pts, r_pts;
    int h = frame.rows, w = frame.cols;
    int yLow = h, yHigh = h / 2;
    int center_offset = 0;
    single_line_offset = 0;

    for (auto& l : lines) {
        double k, b; getLineParams(l, k, b);
        if (k < 0) { l_pts.emplace_back(l[0], l[1]); l_pts.emplace_back(l[2], l[3]); }
        else { r_pts.emplace_back(l[0], l[1]); r_pts.emplace_back(l[2], l[3]); }
    }

    double kl = 0, bl = 0, kr = 0, br = 0;
    bool hasL = false, hasR = false;
    cv::Vec4f lineParam;

    if (l_pts.size() >= 2) {
        cv::fitLine(l_pts, lineParam, cv::DIST_L2, 0, 0.01, 0.01);
        kl = lineParam[1] / lineParam[0];
        bl = lineParam[3] - kl * lineParam[2];
        hasL = true;
    }
    if (r_pts.size() >= 2) {
        cv::fitLine(r_pts, lineParam, cv::DIST_L2, 0, 0.01, 0.01);
        kr = lineParam[1] / lineParam[0];
        br = lineParam[3] - kr * lineParam[2];
        hasR = true;
    }

    if (hasL)
        cv::line(frame, cv::Point((int)((yLow - bl) / kl), yLow), cv::Point((int)((yHigh - bl) / kl), yHigh), cv::Scalar(255, 255, 255), 2);
    if (hasR)
        cv::line(frame, cv::Point((int)((yLow - br) / kr), yLow), cv::Point((int)((yHigh - br) / kr), yHigh), cv::Scalar(255, 255, 255), 2);

    if (lines.size() == 1) {
        double k, b; getLineParams(lines[0], k, b);
        int cx = w / 2;
        cv::line(frame, cv::Point(cx, yLow), cv::Point(cx - (int)((yLow - yHigh) / k), yHigh), cv::Scalar(255, 0, 0), 3);
        single_line_offset = (int)((yLow - b) / k) - cx;
    }
    if (hasL && hasR) {
        int cx1 = ((yLow - bl) / kl + (yLow - br) / kr) / 2;
        int cx2 = ((yHigh - bl) / kl + (yHigh - br) / kr) / 2;
        cv::line(frame, cv::Point(cx1, yLow), cv::Point(cx2, yHigh), cv::Scalar(0, 255, 0), 3);
        center_offset = cx1 - w / 2;
    }
    return center_offset;
}

// ========== 修复 region_of_interest 函数 ==========
void region_of_interest(cv::Mat& img) {
    cv::Mat mask = cv::Mat::zeros(img.size(), img.type());
    std::vector<cv::Point> pts;
    pts.push_back(mask_left_bottom);
    pts.push_back(mask_right_bottom);
    pts.push_back(mask_right_middle);
    pts.push_back(mask_left_middle);
    cv::fillConvexPoly(mask, pts, cv::Scalar(255, 255, 255));
    cv::bitwise_and(img, mask, img);
}

// ==================== YOLO绘制 ====================
static void draw_objects(cv::Mat& img, const std::vector<TargetBox>& boxes) {
    for (auto& box : boxes) {
        int id = box.cate + 1;
        if (id == PERSON_CLASS_ID)
            cv::rectangle(img, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(0, 255, 255), 3);
    }
}

// ==================== MAIN ====================
int main() {
    motor_init(); car_stop();

    yoloF2.init(false);
    yoloF2.loadModel("yolo-fastestv2-opt.param", "yolo-fastestv2-opt.bin");

    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, IMG_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    CarState state = STATE_STOP;
    int green_cnt = 0, red_cnt = 0;
    auto t_start = std::chrono::steady_clock::now();

    while (1) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // ==================== 【1】最高优先级：YOLO检测行人 ====================
        std::vector<TargetBox> boxes;
        yoloF2.detection(frame, boxes);
        draw_objects(frame, boxes);
        bool has_person = false;
        for (auto& b : boxes)
            if (b.cate + 1 == PERSON_CLASS_ID)
                has_person = true;

        if (has_person && !person_detected) {
            person_detected = true;
            person_detect_time = std::chrono::steady_clock::now();
            std::cout << "🚶 检测到人，" << PEDESTRIAN_STOP_DELAY_SEC << "秒后停车" << std::endl;
        }

        bool emergency_stop = false;
        if (person_detected) {
            float dt = std::chrono::duration<float>(std::chrono::steady_clock::now() - person_detect_time).count();
            if (dt >= PEDESTRIAN_STOP_DELAY_SEC) {
                emergency_stop = true;
                car_stop();
                cv::putText(frame, "!!! PERSON STOP !!!", cv::Point(20, 80), 2, 1.2, cv::Scalar(0, 0, 255), 3);
            }
        }

        if (emergency_stop) continue;

        // ==================== 【2】红绿矩形状态机 ====================
        auto rects = detect_color_rect(frame);
        bool has_green = false, has_red = false;
        for (auto& r : rects) {
            if (r.is_green) has_green = true;
            else has_red = true;
        }

        auto now = std::chrono::steady_clock::now();
        int dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start).count();

        if (state == STATE_STOP) {
            car_stop();
            if (has_green) green_cnt++;
            if (dt_ms >= 100) {
                if (green_cnt >= 3) state = STATE_GO;
                green_cnt = 0; t_start = now;
            }
        } else if (state == STATE_GO) {
            // ==================== 【3】车道巡线 ====================
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
            cv::Canny(gray, gray, canny_low_threshold, canny_high_threshold);
            region_of_interest(gray);

            std::vector<cv::Vec4i> lines;
            cv::HoughLinesP(gray, lines, hough_rho, hough_theta, hough_threshold, hough_min_line_length, hough_max_line_gap);
            auto valid = filterAndMergeLines(lines);
            int single_off = 0;
            int offset = drawLanesWithCenterLine(frame, valid, single_off);

            float smooth = 0.85;
            int smooth_err = offset * (1 - smooth) + last_offset * smooth;
            last_offset = smooth_err;
            const int dead = 25;

            if (valid.empty()) car_stop();
            else if (valid.size() == 1) {
                smooth_err = single_off * (1 - smooth) + last_offset * smooth;
                last_offset = smooth_err;
                if (abs(smooth_err) <= dead) car_forward_slow();
                else if (smooth_err < -dead) car_turn_right();
                else car_turn_left();
            } else {
                if (abs(smooth_err) <= dead) car_forward();
                else if (smooth_err < -dead) car_turn_right();
                else car_turn_left();
            }

            if (has_red) red_cnt++;
            if (dt_ms >= 100) {
                if (red_cnt >= 3) { state = STATE_STOP; car_stop(); }
                red_cnt = 0; t_start = now;
            }
        }

        cv::putText(frame, state == STATE_STOP ? "STOP" : "GO", cv::Point(20, 40), 2, 1.5, cv::Scalar(255, 255, 255), 3);
        cv::imshow("CAR", frame);
        if (cv::waitKey(1) == 27) break;
    }

    car_stop();
    cap.release();
    cv::destroyAllWindows();
    pigpio_stop(pi);
    return 0;
}