#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <signal.h>
#include "yolo-fastestv2.h"
#include <pigpiod_if2.h>

// ==================== PWM 电机引脚（你的参数）====================
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

const int PWM_FREQ = 100;       // 已修正：解决电机尖叫
const int DUTY_STRAIGHT = 400000;  // 慢速，能看清线
const int DUTY_LEFT_STR = 220000;
const int DUTY_RIGHT_STR = 140000;

// ==================== 全局安全退出 ====================
bool need_exit = false;
void handle_stop(int sig) {
    need_exit = true;
    std::cout << "\n🛑 CTRL+C 退出，安全停车..." << std::endl;
}

// ==================== 全局变量 ====================
int last_offset = 0;
yoloFastestv2 yoloF2;
int pi;

const unsigned int IMG_WIDTH = 320;
const unsigned int IMG_HEIGHT = 240;

const cv::Point mask_left_bottom(0, IMG_HEIGHT);
const cv::Point mask_right_bottom(IMG_WIDTH-1, IMG_HEIGHT);
const cv::Point mask_left_middle(0, IMG_HEIGHT / 3);   // 扩大ROI，不丢弯道线
const cv::Point mask_right_middle(IMG_WIDTH-1, IMG_HEIGHT / 3);

const unsigned char canny_low_threshold = 60;
const unsigned char canny_high_threshold = 120;

const unsigned char hough_rho = 1;
const double hough_theta = CV_PI / 180;
const unsigned char hough_threshold = 25;
const unsigned char hough_min_line_length = 25;   // 放宽，能识别短线
const unsigned char hough_max_line_gap = 40;

const unsigned char PERSON_CLASS_ID = 1;
const unsigned char BICYCLE_CLASS_ID = 2;
const unsigned char CAR_CLASS_ID = 3;
const unsigned char MOTORBIKE_CLASS_ID = 4;
const unsigned char TRAFFIC_LIGHT_CLASS_ID = 10;
const unsigned char STOP_SIGN_CLASS_ID = 12;

const char* class_names[] = {
    "background", "person", "bicycle", "car", "motorbike",
    "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign"
};

// ==================== 电机初始化 ====================
void motor_init() {
    pi = pigpio_start(NULL, NULL);
    if (pi < 0) {
        std::cerr << "pigpio 连接失败！" << std::endl;
        exit(1);
    }

    set_mode(pi, LF_FWD, PI_OUTPUT);
    set_mode(pi, LF_BWD, PI_OUTPUT);
    set_mode(pi, LB_FWD, PI_OUTPUT);
    set_mode(pi, LB_BWD, PI_OUTPUT);
    set_mode(pi, RF_FWD, PI_OUTPUT);
    set_mode(pi, RF_BWD, PI_OUTPUT);
    set_mode(pi, RB_FWD, PI_OUTPUT);
    set_mode(pi, RB_BWD, PI_OUTPUT);

    set_mode(pi, LF_PWM, PI_OUTPUT);
    set_mode(pi, LB_PWM, PI_OUTPUT);
    set_mode(pi, RF_PWM, PI_OUTPUT);
    set_mode(pi, RB_PWM, PI_OUTPUT);
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

    hardware_PWM(pi, LF_PWM, PWM_FREQ, DUTY_RIGHT_STR);
    hardware_PWM(pi, LB_PWM, PWM_FREQ, DUTY_RIGHT_STR);
    hardware_PWM(pi, RF_PWM, PWM_FREQ, DUTY_LEFT_STR);
    hardware_PWM(pi, RB_PWM, PWM_FREQ, DUTY_LEFT_STR);
}

void car_turn_right() {
    gpio_write(pi, LF_FWD, 1); gpio_write(pi, LF_BWD, 0);
    gpio_write(pi, LB_FWD, 1); gpio_write(pi, LB_BWD, 0);
    gpio_write(pi, RF_FWD, 1); gpio_write(pi, RF_BWD, 0);
    gpio_write(pi, RB_FWD, 1); gpio_write(pi, RB_BWD, 0);

    hardware_PWM(pi, LF_PWM, PWM_FREQ, DUTY_LEFT_STR);
    hardware_PWM(pi, LB_PWM, PWM_FREQ, DUTY_LEFT_STR);
    hardware_PWM(pi, RF_PWM, PWM_FREQ, DUTY_RIGHT_STR);
    hardware_PWM(pi, RB_PWM, PWM_FREQ, DUTY_RIGHT_STR);
}

// ==================== 新版：弯道稳定版车道线拟合 ====================
int drawLanesWithCenterLine(cv::Mat &frame, std::vector<cv::Vec4i> &lines) {
    int h = frame.rows;
    int w = frame.cols;
    int img_center_x = w / 2;
    std::vector<cv::Point> leftPts, rightPts;

    for (auto &l : lines) {
        int x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
        if (x1 == x2) continue;
        double k = (double)(y2 - y1) / (x2 - x1);
        double ang = fabs(atan(k) * 180.0 / CV_PI);

        if (ang < 10 || ang > 80) continue;

        if (k < 0) {
            leftPts.emplace_back(x1, y1);
            leftPts.emplace_back(x2, y2);
        } else {
            rightPts.emplace_back(x1, y1);
            rightPts.emplace_back(x2, y2);
        }
    }

    bool hasLeft = false, hasRight = false;
    float kl = 0, bl = 0, kr = 0, br = 0;
    int yLow = h;
    int yHigh = h / 3;

    if (leftPts.size() >= 2) {
        cv::Vec4f p;
        cv::fitLine(leftPts, p, cv::DIST_L2, 0, 0.01, 0.01);
        kl = p[1] / p[0];
        bl = p[3] - kl * p[2];
        hasLeft = true;
    }
    if (rightPts.size() >= 2) {
        cv::Vec4f p;
        cv::fitLine(rightPts, p, cv::DIST_L2, 0, 0.01, 0.01);
        kr = p[1] / p[0];
        br = p[3] - kr * p[2];
        hasRight = true;
    }

    if (hasLeft) {
        int x1 = (yLow - bl) / kl;
        int x2 = (yHigh - bl) / kl;
        cv::line(frame, cv::Point(x1, yLow), cv::Point(x2, yHigh), cv::Scalar(255,255,255), 2);
    }
    if (hasRight) {
        int x1 = (yLow - br) / kr;
        int x2 = (yHigh - br) / kr;
        cv::line(frame, cv::Point(x1, yLow), cv::Point(x2, yHigh), cv::Scalar(255,255,255), 2);
    }

    int center_offset = 0;
    if (hasLeft && hasRight) {
        int lx = (yLow - bl) / kl;
        int rx = (yLow - br) / kr;
        int cx = (lx + rx) / 2;
        cv::circle(frame, cv::Point(cx, yLow), 5, cv::Scalar(0,255,0), -1);
        cv::line(frame, cv::Point(cx, yLow), cv::Point(cx, yHigh), cv::Scalar(0,255,0), 2);
        cv::line(frame, cv::Point(img_center_x, h), cv::Point(img_center_x, h-60), cv::Scalar(0,0,255), 2);
        center_offset = cx - img_center_x;
    } else if (hasLeft) {
        center_offset = -60;
    } else if (hasRight) {
        center_offset = 60;
    }

    return center_offset;
}

// 废弃原filter，新版已内置
std::vector<cv::Vec4i> filterAndMergeLines(std::vector<cv::Vec4i> &lines) {
    return lines;
}

static void draw_objects(cv::Mat& cvImg, const std::vector<TargetBox>& boxes) {}

void region_of_interest(cv::Mat& img) {
    std::vector<cv::Point> pts = {mask_left_bottom, mask_right_bottom, mask_right_middle, mask_left_middle};
    cv::Mat mask = cv::Mat::zeros(img.size(), img.type());
    cv::fillConvexPoly(mask, pts, cv::Scalar(255));
    cv::bitwise_and(img, mask, img);
}

// ==================== 主函数 ====================
int main(int argc, char** argv) {
    signal(SIGINT, handle_stop);
    motor_init();
    car_stop();

    yoloF2.init(false);
    yoloF2.loadModel("yolo-fastestv2-opt.param", "yolo-fastestv2-opt.bin");

    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    if (!cap.isOpened()) {
        std::cerr << "摄像头打开失败！" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (!need_exit) {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5,5), 0);
        cv::Canny(gray, gray, canny_low_threshold, canny_high_threshold);
        region_of_interest(gray);

        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(gray, lines, hough_rho, hough_theta, hough_threshold,
                        hough_min_line_length, hough_max_line_gap);

        int offset = drawLanesWithCenterLine(frame, lines);

        cv::imshow("edge",gray);
        cv::imshow("lane",frame);
        cv::waitKey(1);








        float smooth = 0.85;
        int s_off = offset*(1-smooth) + last_offset*smooth;
        last_offset = s_off;

        const int dead = 25;
        if (lines.size() < 1) {
            car_stop();
            std::cout << "未检测到车道线 → 停车" << std::endl;
        } else {
            if (abs(s_off) <= dead) {
                car_forward();
                std::cout << "直行 | 偏移: " << s_off << std::endl;
            } else if (s_off < -dead) {
                car_turn_left();
                std::cout << "左转 | 偏移: " << s_off << std::endl;
            } else {
                car_turn_right();
                std::cout << "右转 | 偏移: " << s_off << std::endl;
            }
        }

        std::vector<TargetBox> boxes;
        yoloF2.detection(frame, boxes);
    }

    car_stop();
    cap.release();
    pigpio_stop(pi);
    std::cout << "✅ 程序安全退出" << std::endl;
    return 0;
}