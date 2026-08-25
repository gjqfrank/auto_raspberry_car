#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include "yolo-fastestv2.h"

// #include <gpiod.h>
#include <unistd.h>
#include <pigpiod_if2.h>

// ==================== 电机引脚定义（你的硬件引脚）====================
// L298N motor driver pinout:


constexpr unsigned int PWM_FREQ = 1000; // PWM频率，单位Hz
constexpr unsigned int PWM_DEFAULT = 200000; // 缺省PWM占空比，单位微秒（0-1000000）
constexpr unsigned int PWM_TURN = 230000; // 转弯时PWM占空比，单位微秒（0-1000000）


// struct gpiod_line *rf_fwd, *rf_bwd;
// struct gpiod_line *lf_fwd, *lf_bwd;
// struct gpiod_line *rb_fwd, *rb_bwd;
// struct gpiod_line *lb_fwd, *lb_bwd;
// struct gpiod_chip *chip;

// ==================== 全局平滑变量（防抖核心）====================
int last_offset = 0;

yoloFastestv2 yoloF2;

const unsigned char red_threshold = 200;
const unsigned char green_threshold = 200;
const unsigned char blue_threshold = 200;

const unsigned int IMG_WIDTH = 320;
const unsigned int IMG_HEIGHT = 240;

const cv::Point mask_left_bottom(0, IMG_HEIGHT);
const cv::Point mask_right_bottom(IMG_WIDTH-1, IMG_HEIGHT);
const cv::Point mask_left_middle(0, IMG_HEIGHT/2);
const cv::Point mask_right_middle(IMG_WIDTH-1, IMG_HEIGHT/2);

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

int pi_;

void set_drive_pwm(unsigned int duty_cycle) {
    hardware_PWM(pi_, RF_PWM, PWM_FREQ, duty_cycle);
    hardware_PWM(pi_, LF_PWM, PWM_FREQ, duty_cycle);
    hardware_PWM(pi_, RB_PWM, PWM_FREQ, duty_cycle);
    hardware_PWM(pi_, LB_PWM, PWM_FREQ, duty_cycle);
}

void set_motor_direction(unsigned int fwd_pin, unsigned int bwd_pin, bool forward) {
    gpio_write(pi_, fwd_pin, forward ? 1 : 0);
    gpio_write(pi_, bwd_pin, forward ? 0 : 1);
}

// ==================== 电机控制函数 ====================
void motor_init() {
    // chip = gpiod_chip_open_by_name("gpiochip0");

    // rf_fwd = gpiod_chip_get_line(chip, RF_FWD);
    // rf_bwd = gpiod_chip_get_line(chip, RF_BWD);
    // lf_fwd = gpiod_chip_get_line(chip, LF_FWD);
    // lf_bwd = gpiod_chip_get_line(chip, LF_BWD);
    // rb_fwd = gpiod_chip_get_line(chip, RB_FWD);
    // rb_bwd = gpiod_chip_get_line(chip, RB_BWD);
    // lb_fwd = gpiod_chip_get_line(chip, LB_FWD);
    // lb_bwd = gpiod_chip_get_line(chip, LB_BWD);

    // gpiod_line_request_output(rf_fwd, "motor", 0);
    // gpiod_line_request_output(rf_bwd, "motor", 0);
    // gpiod_line_request_output(lf_fwd, "motor", 0);
    // gpiod_line_request_output(lf_bwd, "motor", 0);
    // gpiod_line_request_output(rb_fwd, "motor", 0);
    // gpiod_line_request_output(rb_bwd, "motor", 0);
    // gpiod_line_request_output(lb_fwd, "motor", 0);
    // gpiod_line_request_output(lb_bwd, "motor", 0);
    pi_ = pigpio_start(NULL, NULL);
    if(pi_ < 0) {
        std::cerr << "Failed to initialize pigpio" << std::endl;
        exit(1);
    }
    set_mode(pi_, RF_FWD, PI_OUTPUT);
    set_mode(pi_, RF_BWD, PI_OUTPUT);
    set_mode(pi_, RB_FWD, PI_OUTPUT);
    set_mode(pi_, RB_BWD, PI_OUTPUT);    
    set_mode(pi_, LF_FWD, PI_OUTPUT);
    set_mode(pi_, LF_BWD, PI_OUTPUT);
    set_mode(pi_, LB_FWD, PI_OUTPUT);
    set_mode(pi_, LB_BWD, PI_OUTPUT);

    set_mode(pi_, RF_PWM, PI_OUTPUT);
    set_mode(pi_, LF_PWM, PI_OUTPUT);
    set_mode(pi_, RB_PWM, PI_OUTPUT);
    set_mode(pi_, LB_PWM, PI_OUTPUT);

    gpio_write(pi_, RF_FWD, 0);
    gpio_write(pi_, RF_BWD, 0);
    gpio_write(pi_, LF_FWD, 0);
    gpio_write(pi_, LF_BWD, 0);
    gpio_write(pi_, RB_FWD, 0);
    gpio_write(pi_, RB_BWD, 0);
    gpio_write(pi_, LB_FWD, 0);
    gpio_write(pi_, LB_BWD, 0); 

    set_drive_pwm(0);

}

// void set_pin(struct gpiod_line* line, int level) {
//     gpiod_line_set_value(line, level);
// }

void car_stop() {
    // set_pin(rf_fwd, 0); set_pin(rf_bwd, 0);
    // set_pin(lf_fwd, 0); set_pin(lf_bwd, 0);
    // set_pin(rb_fwd, 0); set_pin(rb_bwd, 0);
    // set_pin(lb_fwd, 0); set_pin(lb_bwd, 0);

    set_drive_pwm(0);

    gpio_write(pi_, RF_FWD, 0); gpio_write(pi_, RF_BWD, 0);
    gpio_write(pi_, LF_FWD, 0); gpio_write(pi_, LF_BWD, 0);
    gpio_write(pi_, RB_FWD, 0); gpio_write(pi_, RB_BWD, 0);
    gpio_write(pi_, LB_FWD, 0); gpio_write(pi_, LB_BWD, 0);
}

void car_forward() {
    // set_pin(rf_fwd, 1); set_pin(rf_bwd, 0);
    // set_pin(lf_fwd, 1); set_pin(lf_bwd, 0);
    // set_pin(rb_fwd, 1); set_pin(rb_bwd, 0);
    // set_pin(lb_fwd, 1); set_pin(lb_bwd, 0);

    set_motor_direction(RF_FWD, RF_BWD, true);
    set_motor_direction(LF_FWD, LF_BWD, true);
    set_motor_direction(RB_FWD, RB_BWD, true);
    set_motor_direction(LB_FWD, LB_BWD, true);
    set_drive_pwm(PWM_DEFAULT);
}

void car_turn_left() {
    // set_pin(rf_fwd, 1); set_pin(rf_bwd, 0);
    // set_pin(lf_fwd, 0); set_pin(lf_bwd, 0);
    // set_pin(rb_fwd, 1); set_pin(rb_bwd, 0);
    // set_pin(lb_fwd, 0); set_pin(lb_bwd, 0);

    set_motor_direction(RF_FWD, RF_BWD, true);
    set_motor_direction(RB_FWD, RB_BWD, true);
    set_motor_direction(LF_FWD, LF_BWD, true);
    set_motor_direction(LB_FWD, LB_BWD, true);

    hardware_PWM(pi_, RF_PWM, PWM_FREQ, PWM_TURN);
    hardware_PWM(pi_, RB_PWM, PWM_FREQ, PWM_TURN);
    hardware_PWM(pi_, LF_PWM, PWM_FREQ, PWM_DEFAULT);
    hardware_PWM(pi_, LB_PWM, PWM_FREQ, PWM_DEFAULT);
}

void car_turn_right() {
    // set_pin(rf_fwd, 0); set_pin(rf_bwd, 0);
    // set_pin(lf_fwd, 1); set_pin(lf_bwd, 0);
    // set_pin(rb_fwd, 0); set_pin(rb_bwd, 0);
    // set_pin(lb_fwd, 1); set_pin(lb_bwd, 0);

    set_motor_direction(RF_FWD, RF_BWD, true);
    set_motor_direction(RB_FWD, RB_BWD, true);
    set_motor_direction(LF_FWD, LF_BWD, true);
    set_motor_direction(LB_FWD, LB_BWD, true);

    hardware_PWM(pi_, RF_PWM, PWM_FREQ, PWM_DEFAULT);
    hardware_PWM(pi_, RB_PWM, PWM_FREQ, PWM_DEFAULT);
    hardware_PWM(pi_, LF_PWM, PWM_FREQ, PWM_TURN);
    hardware_PWM(pi_, LB_PWM, PWM_FREQ, PWM_TURN);
}

// ==================== 车道线工具函数 ====================
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
        double ang = fabs(atan(k) * 180 / CV_PI);

        if (ang < 20 || ang > 75) continue;

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
            leftPts.push_back(p2);
        } else {
            rightPts.push_back(p1);
            rightPts.push_back(p2);
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
        int cx1 = ((yLow - bl) / kl + (yLow - br) / kr) / 2;
        int cx2 = ((yHigh - bl) / kl + (yHigh - br) / kr) / 2;
        cv::line(frame, cv::Point(cx1, yLow), cv::Point(cx2, yHigh), cv::Scalar(0,255,0), 3);

        int img_center_x = w / 2;
        center_offset = cx1 - img_center_x;
        cv::line(frame, cv::Point(img_center_x, h), cv::Point(img_center_x, h-30), cv::Scalar(0,0,255),2);
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
        std::cerr << "Can not open camera" << std::endl;
        return -1;
    }

    cv::Mat frame;
    float fpsWindow[16];
    int fpsIdx = 0;
    for (int i = 0; i < 16; i++) fpsWindow[i] = 0.0f;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Can not read frame" << std::endl;
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

        // ==================== 防抖平滑核心 ====================
        float smooth_factor = 0.85;
        int smooth_offset = (offset * (1 - smooth_factor)) + (last_offset * smooth_factor);
        last_offset = smooth_offset;

        // ==================== 循迹控制 ====================
        const int dead_zone = 25;

        if (validLines.size() < 2) {
            car_stop();
        } else {
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

        cv::putText(frameCopy, cv::format("line num: %zu", lines.size()),
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);

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

    cap.release();
    cv::destroyAllWindows();
    pigpio_stop(pi_);
    return 0;
}