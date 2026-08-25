#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include "yolo-fastestv2.h"

// ==================== PWM 驱动库 ====================
#include <pigpiod_if2.h>

// ==================== 电机引脚定义（你的PWM硬件引脚）====================
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

// ==================== PWM 参数（你的测试参数）====================
const int PWM_FREQ = 100;
const int DUTY_STRAIGHT = 130000;   // 直行基准占空比
const int DUTY_LEFT_STR = 300000;  // 左轮强
const int DUTY_RIGHT_STR = 130000; // 右轮弱
const int DUTY_SINGLE_LINE = 130000; // 单车道线时的基础占空比（降低速度，避免失控）

// ==================== 全局平滑变量 ====================
int last_offset = 0;
int last_single_line_dir = 0; // 上一次单车道线的转向方向：-1=左，0=无，1=右

yoloFastestv2 yoloF2;

const unsigned char red_threshold = 200;
const unsigned char green_threshold = 200;
const unsigned char blue_threshold = 200;

const unsigned int IMG_WIDTH = 320;
const unsigned int IMG_HEIGHT = 240;

const cv::Point mask_left_bottom(0, IMG_HEIGHT);
const cv::Point mask_right_bottom(IMG_WIDTH-1, IMG_HEIGHT);
const cv::Point mask_left_middle(0, IMG_HEIGHT*3/4);
const cv::Point mask_right_middle(IMG_WIDTH-1, IMG_HEIGHT*3/4);

const unsigned char canny_low_threshold = 60;
const unsigned char canny_high_threshold = 120;

const unsigned char hough_rho = 1;
const double hough_theta = CV_PI / 180;
const unsigned char hough_threshold = 35;
const unsigned char hough_min_line_length = 50;
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

// ==================== PWM 全局句柄 ====================
int pi;

// ==================== PWM 电机初始化 ====================
void motor_init() {
    pi = pigpio_start(NULL, NULL);
    if (pi < 0) {
        std::cerr << "pigpio 连接失败！" << std::endl;
        exit(1);
    }

    // 方向引脚
    set_mode(pi, LF_FWD, PI_OUTPUT);
    set_mode(pi, LF_BWD, PI_OUTPUT);
    set_mode(pi, LB_FWD, PI_OUTPUT);
    set_mode(pi, LB_BWD, PI_OUTPUT);
    set_mode(pi, RF_FWD, PI_OUTPUT);
    set_mode(pi, RF_BWD, PI_OUTPUT);
    set_mode(pi, RB_FWD, PI_OUTPUT);
    set_mode(pi, RB_BWD, PI_OUTPUT);

    // PWM引脚
    set_mode(pi, LF_PWM, PI_OUTPUT);
    set_mode(pi, LB_PWM, PI_OUTPUT);
    set_mode(pi, RF_PWM, PI_OUTPUT);
    set_mode(pi, RB_PWM, PI_OUTPUT);
}

// ==================== PWM 电机动作函数 ====================
void car_stop() {
    // 方向清零
    gpio_write(pi, LF_FWD, 0); gpio_write(pi, LF_BWD, 0);
    gpio_write(pi, LB_FWD, 0); gpio_write(pi, LB_BWD, 0);
    gpio_write(pi, RF_FWD, 0); gpio_write(pi, RF_BWD, 0);
    gpio_write(pi, RB_FWD, 0); gpio_write(pi, RB_BWD, 0);

    // PWM停止
    hardware_PWM(pi, LF_PWM, 0, 0);
    hardware_PWM(pi, LB_PWM, 0, 0);
    hardware_PWM(pi, RF_PWM, 0, 0);
    hardware_PWM(pi, RB_PWM, 0, 0);
}

void car_forward() {
    // 方向：全部前进
    gpio_write(pi, LF_FWD, 1); gpio_write(pi, LF_BWD, 0);
    gpio_write(pi, LB_FWD, 1); gpio_write(pi, LB_BWD, 0);
    gpio_write(pi, RF_FWD, 1); gpio_write(pi, RF_BWD, 0);
    gpio_write(pi, RB_FWD, 1); gpio_write(pi, RB_BWD, 0);

    // 左右同速
    hardware_PWM(pi, LF_PWM, PWM_FREQ, DUTY_STRAIGHT);
    hardware_PWM(pi, LB_PWM, PWM_FREQ, DUTY_STRAIGHT);
    hardware_PWM(pi, RF_PWM, PWM_FREQ, DUTY_STRAIGHT);
    hardware_PWM(pi, RB_PWM, PWM_FREQ, DUTY_STRAIGHT);
}

void car_turn_left() {
    // 方向：全部前进
    gpio_write(pi, LF_FWD, 1); gpio_write(pi, LF_BWD, 0);
    gpio_write(pi, LB_FWD, 1); gpio_write(pi, LB_BWD, 0);
    gpio_write(pi, RF_FWD, 1); gpio_write(pi, RF_BWD, 0);
    gpio_write(pi, RB_FWD, 1); gpio_write(pi, RB_BWD, 0);

    // 左轮快、右轮慢（差速左转）
    hardware_PWM(pi, LF_PWM, PWM_FREQ, DUTY_LEFT_STR);
    hardware_PWM(pi, LB_PWM, PWM_FREQ, DUTY_LEFT_STR);
    hardware_PWM(pi, RF_PWM, PWM_FREQ, DUTY_RIGHT_STR);
    hardware_PWM(pi, RB_PWM, PWM_FREQ, DUTY_RIGHT_STR);
}

void car_turn_right() {
    // 方向：全部前进
    gpio_write(pi, LF_FWD, 1); gpio_write(pi, LF_BWD, 0);
    gpio_write(pi, LB_FWD, 1); gpio_write(pi, LB_BWD, 0);
    gpio_write(pi, RF_FWD, 1); gpio_write(pi, RF_BWD, 0);
    gpio_write(pi, RB_FWD, 1); gpio_write(pi, RB_BWD, 0);

    // 右轮快、左轮慢（差速右转）
    hardware_PWM(pi, LF_PWM, PWM_FREQ, DUTY_RIGHT_STR);
    hardware_PWM(pi, LB_PWM, PWM_FREQ, DUTY_RIGHT_STR);
    hardware_PWM(pi, RF_PWM, PWM_FREQ, DUTY_LEFT_STR);
    hardware_PWM(pi, RB_PWM, PWM_FREQ, DUTY_LEFT_STR);
}

// 单车道线时的精细转向（自定义占空比，控制转向幅度）
void car_turn_single_line(int left_duty, int right_duty) {
    // 方向：全部前进
    gpio_write(pi, LF_FWD, 1); gpio_write(pi, LF_BWD, 0);
    gpio_write(pi, LB_FWD, 1); gpio_write(pi, LB_BWD, 0);
    gpio_write(pi, RF_FWD, 1); gpio_write(pi, RF_BWD, 0);
    gpio_write(pi, RB_FWD, 1); gpio_write(pi, RB_BWD, 0);

    // 自定义左右轮占空比，实现小幅转向
    hardware_PWM(pi, LF_PWM, PWM_FREQ, left_duty);
    hardware_PWM(pi, LB_PWM, PWM_FREQ, left_duty);
    hardware_PWM(pi, RF_PWM, PWM_FREQ, right_duty);
    hardware_PWM(pi, RB_PWM, PWM_FREQ, right_duty);
}

// ==================== 车道线工具函数（不变）====================
void getLineParams(cv::Vec4i line, double &k, double &b) {
    int x1 = line[0], y1 = line[1];
    int x2 = line[2], y2 = line[3];
    if (x2 == x1) {
        k = 1e6; // 垂直直线，斜率设为极大值
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
        double ang = fabs(atan(k) * 180 / CV_PI);

        // if (ang < 20 || ang > 75) continue; // 过滤水平/接近垂直的无效线
        if (ang <10) continue;

        bool dup = false;
        for (auto &kb : kb_list) {
            if (fabs(k - kb.first) < 0.18 && fabs(b - kb.second) < 50) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            valid.push_back(l);
            kb_list.emplace_back(k, b);
        }
    }
    return valid;
}

int drawLanesWithCenterLine(cv::Mat &frame, std::vector<cv::Vec4i> &lines) {
    std::vector<cv::Point> leftPts, rightPts;
    int h = frame.rows;
    int w = frame.cols;
    int yLow = h;
    int yHigh = h / 2;

    for (auto &l : lines) {
        double k, b;
        getLineParams(l, k, b);
        cv::Point p1(l[0], l[1]), p2(l[2], l[3]);
        if (k < 0) {
            leftPts.push_back(p1);
            leftPts.push_back(p2); // 斜率为负 → 左侧车道线
        } else {
            rightPts.push_back(p1);
            rightPts.push_back(p2); // 斜率为正 → 右侧车道线
        }
    }

    double kl = 0, bl = 0, kr = 0, br = 0;
    bool hasLeft = false, hasRight = false;
    int center_offset = 0;

    if (leftPts.size() >= 2) {
        cv::Vec4f lineParam;
        cv::fitLine(leftPts, lineParam, cv::DIST_L2, 0, 0.01, 0.01);
        double vx = lineParam[0], vy = lineParam[1], x0 = lineParam[2], y0 = lineParam[3];
        kl = vy / vx;
        bl = y0 - kl * x0;
        hasLeft = true;
    }

    if (rightPts.size() >= 2) {
        cv::Vec4f lineParam;
        cv::fitLine(rightPts, lineParam, cv::DIST_L2, 0, 0.01, 0.01);
        double vx = lineParam[0], vy = lineParam[1], x0 = lineParam[2], y0 = lineParam[3];
        kr = vy / vx;
        br = y0 - kr * x0;
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

    if (hasLeft && hasRight) {
        // 双车道线 → 计算中心偏移
        int cx1 = ((yLow - bl) / kl + (yLow - br) / kr) / 2;
        int cx2 = ((yHigh - bl) / kl + (yHigh - br) / kr) / 2;
        cv::line(frame, cv::Point(cx1, yLow), cv::Point(cx2, yHigh), cv::Scalar(0,255,0), 3);

        int img_center_x = w / 2;
        center_offset = cx1 - img_center_x;
        cv::line(frame, cv::Point(img_center_x, h), cv::Point(img_center_x, h-30), cv::Scalar(0,0,255),2);
    } else if (hasLeft || hasRight) {
        // 单车道线 → 计算该线相对于中心的偏移
        int line_x_at_bottom = 0;
        if (hasLeft) {
            line_x_at_bottom = (yLow - bl) / kl;
        } else {
            line_x_at_bottom = (yLow - br) / kr;
        }
        int img_center_x = w / 2;
        center_offset = line_x_at_bottom - img_center_x;
        // 绘制单车道线的参考线
        cv::line(frame, cv::Point(img_center_x, h), cv::Point(img_center_x, h-30), cv::Scalar(0,0,255),2);
    }

    return center_offset;
}

// 分析单车道线的转向策略（返回：-1=左转，1=右转，0=直行）
int analyzeSingleLineSteering(std::vector<cv::Vec4i> &validLines, int offset, int &left_duty, int &right_duty) {
    double k, b;
    getLineParams(validLines[0], k, b);
    int steering_dir = 0;
    int img_center_x = IMG_WIDTH / 2;
    int offset_abs = abs(offset);

    // 斜率<0 → 左侧车道线；斜率>0 → 右侧车道线
    if (k < 0) { 
        // 左侧车道线：
        // - 线在中心左侧（offset<0）→ 小车偏左，需要右转
        // - 线在中心右侧（offset>0）→ 小车偏右，需要左转（但这种情况极少）
        if (offset < 0) {
            steering_dir = 1; // 右转
            // 偏移越大，右转幅度越大（右轮占空比越高）
            right_duty = DUTY_SINGLE_LINE + (offset_abs * 500); // 动态调整右轮占空比
            left_duty = DUTY_SINGLE_LINE - (offset_abs * 300);
        } else {
            steering_dir = -1; // 左转
            left_duty = DUTY_SINGLE_LINE + (offset_abs * 500);
            right_duty = DUTY_SINGLE_LINE - (offset_abs * 300);
        }
    } else { 
        // 右侧车道线：
        // - 线在中心右侧（offset>0）→ 小车偏右，需要左转
        // - 线在中心左侧（offset<0）→ 小车偏左，需要右转（但这种情况极少）
        if (offset > 0) {
            steering_dir = -1; // 左转
            left_duty = DUTY_SINGLE_LINE + (offset_abs * 500);
            right_duty = DUTY_SINGLE_LINE - (offset_abs * 300);
        } else {
            steering_dir = 1; // 右转
            right_duty = DUTY_SINGLE_LINE + (offset_abs * 500);
            left_duty = DUTY_SINGLE_LINE - (offset_abs * 300);
        }
    }

    // 限制占空比范围，避免超出硬件阈值
    left_duty = std::max(300000, std::min(500000, left_duty));
    right_duty = std::max(300000, std::min(500000, right_duty));

    return steering_dir;
}

static void draw_objects(cv::Mat& cvImg, const std::vector<TargetBox>& boxes) {
    unsigned char id;
    for (size_t i = 0; i < boxes.size(); i++) {
        char text[256];
        sprintf(text, "%s %.1f%%", class_names[boxes[i].cate + 1], boxes[i].score * 100);
        id = boxes[i].cate + 1;

        if (id != PERSON_CLASS_ID && id != BICYCLE_CLASS_ID && id != CAR_CLASS_ID &&
            id != MOTORBIKE_CLASS_ID && id != TRAFFIC_LIGHT_CLASS_ID && id != STOP_SIGN_CLASS_ID) {
            continue;
        }

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
    std::vector<cv::Point> vertices;
    vertices.push_back(mask_left_bottom);
    vertices.push_back(mask_right_bottom);
    vertices.push_back(mask_right_middle);
    vertices.push_back(mask_left_middle);
    cv::Mat mask = cv::Mat::zeros(img.size(), img.type());
    cv::fillConvexPoly(mask, vertices, cv::Scalar(255, 255, 255));
    cv::bitwise_and(img, mask, img);
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
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    if (!cap.isOpened()) {
        std::cerr << "摄像头打开失败！" << std::endl;
        return -1;
    }

    cv::Mat frame;
    float fpsWindow[16];
    int fpsIdx = 0;
    for (int i = 0; i < 16; i++) fpsWindow[i] = 0.0f;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "读取帧失败！" << std::endl;
            break;
        }

        cv::Mat frameCopy = frame.clone();
        cv::cvtColor(frameCopy, frameCopy, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(frameCopy, frameCopy, cv::Size(5, 5), 0);
        cv::Canny(frameCopy, frameCopy, canny_low_threshold, canny_high_threshold);
        region_of_interest(frameCopy);

        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(frameCopy, lines, hough_rho, hough_theta, hough_threshold,
                        hough_min_line_length, hough_max_line_gap);

        auto validLines = filterAndMergeLines(lines);
        int offset = drawLanesWithCenterLine(frame, validLines);

        // 平滑防抖
        float smooth_factor = 0.85;
        int smooth_offset = (offset * (1 - smooth_factor)) + (last_offset * smooth_factor);
        last_offset = smooth_offset;

        // 循迹控制逻辑
        const int dead_zone = 25;
        int left_duty = DUTY_SINGLE_LINE, right_duty = DUTY_SINGLE_LINE;
        int single_line_dir = 0;

        if (validLines.size() == 0) {
            // 无车道线 → 停车
            car_stop();
            std::cout << "未检测到车道线 → 停车" << std::endl;
        } else if (validLines.size() == 1) {
            // 单车道线 → 分析斜率并转向
            single_line_dir = analyzeSingleLineSteering(validLines, smooth_offset, left_duty, right_duty);
            // 平滑上一次的转向方向，避免抖动
            single_line_dir = (single_line_dir * 0.7) + (last_single_line_dir * 0.3);
            last_single_line_dir = single_line_dir;
            
            car_turn_single_line(left_duty, right_duty);
            std::cout << "单车道线 | 斜率: " << [&]() {
                double k, b; getLineParams(validLines[0], k, b); return k;
            }() << " | 偏移: " << smooth_offset << " | 转向: " 
                << (single_line_dir < 0 ? "左转" : (single_line_dir > 0 ? "右转" : "直行")) << std::endl;
        } else {
            // 双车道线 → 原逻辑
            last_single_line_dir = 0; // 重置单车道线转向记录
            if (abs(smooth_offset) <= dead_zone) {
                car_forward();
                std::cout << "直行 | 偏移: " << smooth_offset << std::endl;
            } else if (smooth_offset < -dead_zone) {
                car_turn_left();
                std::cout << "左转 | 偏移: " << smooth_offset << std::endl;
            } else {
                car_turn_right();
                std::cout << "右转 | 偏移: " << smooth_offset << std::endl;
            }
        }

        cv::putText(frameCopy, cv::format("line num: %d", lines.size()),
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);

        // YOLO检测
        const auto tBegin = std::chrono::steady_clock::now();
        std::vector<TargetBox> boxes;
        yoloF2.detection(frame, boxes);
        draw_objects(frame, boxes);
        const auto tEnd = std::chrono::steady_clock::now();
        float ms = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tBegin).count();
        if (ms > 0.0f) fpsWindow[(fpsIdx++) & 0x0F] = 1000.0f / ms;
        float fps = 0.0f;
        for (int i = 0; i < 16; i++) fps += fpsWindow[i];
        fps /= 16.0f;

        cv::putText(frame, cv::format("FPS %0.2f", fps), cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255));

        cv::imshow("Camera Lane Detection", frameCopy);
        cv::imshow("Original", frame);

        if (cv::waitKey(5) == 27) {
            car_stop();
            break;
        }
    }

    // 释放资源
    cap.release();
    cv::destroyAllWindows();
    car_stop();
    pigpio_stop(pi);

    return 0;
}