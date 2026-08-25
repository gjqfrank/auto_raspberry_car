#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include "yolo-fastestv2.h"
#include <pigpiod_if2.h>

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
const int DUTY_STRAIGHT = 200000;
const int DUTY_SLOW = 170000;
const int DUTY_LEFT_STR = 330000;
const int DUTY_RIGHT_STR = 170000;

int last_offset = 0;
yoloFastestv2 yoloF2;

const unsigned int IMG_WIDTH = 320;
const unsigned int IMG_HEIGHT = 240;

const cv::Point mask_left_bottom(0, IMG_HEIGHT);
const cv::Point mask_right_bottom(IMG_WIDTH-1, IMG_HEIGHT);
const cv::Point mask_left_middle(0, IMG_HEIGHT*2/3);
const cv::Point mask_right_middle(IMG_WIDTH-1, IMG_HEIGHT*2/3);

const unsigned char canny_low_threshold = 60;
const unsigned char canny_high_threshold = 120;

const unsigned char hough_rho = 1;
const double hough_theta = CV_PI / 180;
const unsigned char hough_threshold = 35;
const unsigned char hough_min_line_length = 20;
const unsigned char hough_max_line_gap = 40;

const unsigned char PERSON_CLASS_ID = 1;
const unsigned char BICYCLE_CLASS_ID = 2;
const unsigned char CAR_CLASS_ID = 3;
const unsigned char MOTORBIKE_CLASS_ID = 4;
const unsigned char TRAFFIC_LIGHT_CLASS_ID = 10;
const unsigned char STOP_SIGN_CLASS_ID = 12;

const char* class_names[] = {
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
};

int pi;

// ==================== 电机初始化 ====================
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
    hardware_PWM(pi, LF_PWM, 0, 0); hardware_PWM(pi, LB_PWM, 0, 0);
    hardware_PWM(pi, RF_PWM, 0, 0); hardware_PWM(pi, RB_PWM, 0, 0);
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
    gpio_write(pi, LF_FWD, 0); gpio_write(pi, LF_BWD, 0);
    gpio_write(pi, LB_FWD, 0); gpio_write(pi, LB_BWD, 0);
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
    gpio_write(pi, RF_FWD, 0); gpio_write(pi, RF_BWD, 0);
    gpio_write(pi, RB_FWD, 0); gpio_write(pi, RB_BWD, 0);
    hardware_PWM(pi, LF_PWM, PWM_FREQ, DUTY_RIGHT_STR);
    hardware_PWM(pi, LB_PWM, PWM_FREQ, DUTY_RIGHT_STR);
    hardware_PWM(pi, RF_PWM, PWM_FREQ, DUTY_LEFT_STR);
    hardware_PWM(pi, RB_PWM, PWM_FREQ, DUTY_LEFT_STR);
}

void getLineParams(cv::Vec4i line, double &k, double &b) {
    int x1 = line[0], y1 = line[1];
    int x2 = line[2], y2 = line[3];
    if (x2 == x1) {
        k = 1e6;
    } else {
        k = (double)(y2 - y1) / (x2 - x1);
    }
    b = y1 - k * x1;
}

std::vector<cv::Vec4i> filterAndMergeLines(std::vector<cv::Vec4i> &lines) {
    std::vector<cv::Vec4i> valid;
    std::vector<std::pair<double, double>> kb_list;
    for (auto &l : lines) {
        double k, b;
        getLineParams(l, k, b);
        if (fabs(k) < 0.1) continue;
        double ang = fabs(atan(k) * 180 / CV_PI);
        if (ang < 20) continue;
        bool dup = false;
        for (auto &kb : kb_list) {
            if (fabs(k - kb.first) < 0.18 && fabs(b - kb.second) < 50) {
                dup = true; break;
            }
        }
        if (!dup) {
            valid.push_back(l);
            kb_list.emplace_back(k, b);
        }
    }
    return valid;
}

// ==================== 统一绘图函数：蓝线 + 绿线 算法一致 ====================
int drawLanesWithCenterLine(cv::Mat &frame, std::vector<cv::Vec4i> &lines, int &single_line_offset) {
    std::vector<cv::Point> leftPts, rightPts;
    int h = frame.rows;
    int w = frame.cols;
    int yLow = h;
    int yHigh = h / 2;
    int center_offset = 0;
    single_line_offset = 0;

    for (auto &l : lines) {
        double k, b;
        getLineParams(l, k, b);
        cv::Point p1(l[0], l[1]), p2(l[2], l[3]);
        if (k < 0) { leftPts.push_back(p1); leftPts.push_back(p2); }
        else { rightPts.push_back(p1); rightPts.push_back(p2); }
    }

    double kl=0, bl=0, kr=0, br=0;
    bool hasLeft=false, hasRight=false;

    if (leftPts.size() >= 2) {
        cv::Vec4f lp;
        cv::fitLine(leftPts, lp, cv::DIST_L2, 0, 0.01, 0.01);
        kl = lp[1]/lp[0]; bl = lp[3] - kl*lp[2]; hasLeft = true;
    }
    if (rightPts.size() >= 2) {
        cv::Vec4f rp;
        cv::fitLine(rightPts, rp, cv::DIST_L2, 0, 0.01, 0.01);
        kr = rp[1]/rp[0]; br = rp[3] - kr*rp[2]; hasRight = true;
    }

    if (hasLeft) {
        int x1 = (yLow - bl)/kl;
        int x2 = (yHigh - bl)/kl;
        cv::line(frame, cv::Point(x1,yLow), cv::Point(x2,yHigh), cv::Scalar(255,255,255), 2);
    }
    if (hasRight) {
        int x1 = (yLow - br)/kr;
        int x2 = (yHigh - br)/kr;
        cv::line(frame, cv::Point(x1,yLow), cv::Point(x2,yHigh), cv::Scalar(255,255,255), 2);
    }

    // 单条线 → 蓝色中心线（和绿线同算法）
    if (lines.size() == 1) {
        double k, b;
        getLineParams(lines[0], k, b);
        int cx = w / 2;
        int x1 = cx;
        int y1 = yLow;
        int x2 = x1 - (y1 - yHigh) / k;
        int y2 = yHigh;
        cv::line(frame, cv::Point(x1,y1), cv::Point(x2,y2), cv::Scalar(255,0,0), 3);

        int target_x = (yLow - b) / k;
        single_line_offset = target_x - cx;
    }

    // 双条线 → 绿色中心线
    if (hasLeft && hasRight) {
        int cx1 = ((yLow-bl)/kl + (yLow-br)/kr) / 2;
        int cx2 = ((yHigh-bl)/kl + (yHigh-br)/kr) / 2;
        cv::line(frame, cv::Point(cx1,yLow), cv::Point(cx2,yHigh), cv::Scalar(0,255,0), 3);
        center_offset = cx1 - w/2;
    }

    return center_offset;
}

static void draw_objects(cv::Mat& cvImg, const std::vector<TargetBox>& boxes) {
    unsigned char id;
    for (size_t i = 0; i < boxes.size(); i++) {
        char text[256];
        sprintf(text, "%s %.1f%%", class_names[boxes[i].cate + 1], boxes[i].score * 100);
        id = boxes[i].cate + 1;
        if (id != PERSON_CLASS_ID && id != BICYCLE_CLASS_ID && id != CAR_CLASS_ID &&
            id != MOTORBIKE_CLASS_ID && id != TRAFFIC_LIGHT_CLASS_ID && id != STOP_SIGN_CLASS_ID)
            continue;
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int x = boxes[i].x1;
        int y = boxes[i].y1 - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > cvImg.cols) x = cvImg.cols - label_size.width;
        cv::rectangle(cvImg, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);
        cv::putText(cvImg, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        cv::rectangle(cvImg, cv::Point(boxes[i].x1, boxes[i].y1),
                      cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255, 0, 0));
    }
}

void region_of_interest(cv::Mat& img) {
    std::vector<cv::Point> v{mask_left_bottom, mask_right_bottom, mask_right_middle, mask_left_middle};
    cv::Mat m = cv::Mat::zeros(img.size(), img.type());
    cv::fillConvexPoly(m, v, cv::Scalar(255,255,255));
    cv::bitwise_and(img, m, img);
}

// ==================== 主函数 ====================
int main(int argc, char** argv) {
    motor_init();
    car_stop();

    yoloF2.init(false);
    yoloF2.loadModel("yolo-fastestv2-opt.param", "yolo-fastestv2-opt.bin");

    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));

    if (!cap.isOpened()) {
        std::cerr << "摄像头打开失败！" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat gray = frame.clone();
        // 修复：添加 cv:: 前缀
        cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5,5), 0);
        cv::Canny(gray, gray, canny_low_threshold, canny_high_threshold);
        region_of_interest(gray);

        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(gray, lines, hough_rho, hough_theta, hough_threshold,
                    hough_min_line_length, hough_max_line_gap);
        auto valid = filterAndMergeLines(lines);

        int control_err = 0;
        int single_offset = 0;

        // ==================== 统一控制逻辑 ====================
        if (valid.size() == 0) {
            car_forward();
            std::cout << "无车道线 → 停车" << std::endl;
        }
        else {
            int dual_offset = drawLanesWithCenterLine(frame, valid, single_offset);

            if (valid.size() == 1) {
                control_err = single_offset;
                std::cout << "沿蓝线行驶 | 偏移:" << control_err << std::endl;
            } else {
                control_err = dual_offset;
                std::cout << "沿绿线行驶 | 偏移:" << control_err << std::endl;
            }

            // 平滑
            float smooth = 0.85;
            int smooth_err = control_err * (1 - smooth) + last_offset * smooth;
            last_offset = smooth_err;

            const int dead = 25;

            // 单条线 → 减速
            if (valid.size() == 1) {
                if (abs(smooth_err) <= dead) {
                    car_forward_slow();
                } else if (smooth_err < -dead) {
                    car_turn_right();
                } else {
                    car_turn_left();
                }
            }
            // 两条线 → 正常
            else {
                if (abs(smooth_err) <= dead) {
                    car_forward();
                } else if (smooth_err < -dead) {
                    car_turn_right();
                } else {
                    car_turn_left();
                }
            }
        }

        cv::imshow("edge", gray);
        cv::imshow("lane", frame);
        if (cv::waitKey(10) == 27) { car_stop(); break; }
    }

    cap.release();
    cv::destroyAllWindows();
    car_stop();
    pigpio_stop(pi);
    return 0;
}