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

// -------------------------- Motor Pin Definitions --------------------------
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

// -------------------------- PWM Parameters --------------------------
const int PWM_FREQ = 100;
const int DUTY_STRAIGHT = 140000;
const int DUTY_LEFT_STR = 290000;
const int DUTY_RIGHT_STR = 130000;

// const int PWM_FREQ = 0;
//const int DUTY_STRAIGHT = 0;
//const int DUTY_LEFT_STR = 0;
//const int DUTY_RIGHT_STR = 0;

// -------------------------- Image Parameters --------------------------
const unsigned int IMG_WIDTH = 320;
const unsigned int IMG_HEIGHT = 240;
const int dead_zone = 25;

// -------------------------- Global Variables --------------------------
int pi;
int last_offset = 0;
float last_valid_slope = 0.0f;
yoloFastestv2 yoloF2;

// -------------------------- Motor Control Functions --------------------------
void motor_init() {
    pi = pigpio_start(NULL, NULL);
    if (pi < 0) {
        std::cerr << "pigpio initialization failed!" << std::endl;
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

// -------------------------- Lane Detection Helper Functions --------------------------
std::pair<float, float> getLineParams(cv::Vec4i line) {
    int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
    if (x1 == x2) return {1000.0f, 0.0f};
    float k = (float)(y2 - y1) / (x2 - x1);
    float b = y1 - k * x1;
    return {k, b};
}

std::vector<cv::Vec4i> filterAndMergeLines(std::vector<cv::Vec4i>& lines) {
    std::vector<cv::Vec4i> valid;
    for (auto& line : lines) {
        float k = getLineParams(line).first;
        if (fabs(k) > 0.2 && fabs(k) < 3.0) {
            valid.push_back(line);
        }
    }
    return valid;
}

void region_of_interest(cv::Mat& img) {
    cv::Mat mask = cv::Mat::zeros(img.size(), img.type());
    cv::Point pts[4] = {
        cv::Point(0, IMG_HEIGHT/2),
        cv::Point(0, IMG_HEIGHT),
        cv::Point(IMG_WIDTH, IMG_HEIGHT),
        cv::Point(IMG_WIDTH, IMG_HEIGHT/2)
    };
    cv::fillConvexPoly(mask, pts, 4, cv::Scalar(255));
    img &= mask;
}

int drawLanesWithCenterLine(cv::Mat& img, std::vector<cv::Vec4i>& lines) {
    std::vector<float> left_ks, right_ks;
    int center_x = IMG_WIDTH / 2;

    for (auto& line : lines) {
        float k = getLineParams(line).first;
        if (k < 0) left_ks.push_back(k);
        else right_ks.push_back(k);

        // -------------------------- Draw slope value next to the line --------------------------
        char slope_text[32];
        sprintf(slope_text, "k=%.2f", k);
        cv::putText(img, slope_text,
                    cv::Point(line[0]+5, line[1]+15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    cv::Scalar(0, 255, 255), 1);

        cv::line(img, cv::Point(line[0], line[1]),
                 cv::Point(line[2], line[3]),
                 cv::Scalar(0, 255, 0), 2);
    }

    int target_x = center_x;
    if (!left_ks.empty() && !right_ks.empty()) {
        target_x = center_x;
    } else if (!left_ks.empty()) {
        target_x = IMG_WIDTH * 0.75;
    } else if (!right_ks.empty()) {
        target_x = IMG_WIDTH * 0.25;
    }

    cv::circle(img, cv::Point(target_x, IMG_HEIGHT-20), 5, cv::Scalar(0,0,255), -1);
    return target_x - center_x;
}

// -------------------------- Main Function --------------------------
int main() {
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, IMG_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT);
    motor_init();
    car_stop();

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat gray, blur, edge;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blur, cv::Size(5,5), 0);
        cv::Canny(blur, edge, 60, 120);
        region_of_interest(edge);

        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edge, lines, 1, CV_PI/180, 35, 50, 40);
        auto validLines = filterAndMergeLines(lines);

        int offset = drawLanesWithCenterLine(frame, validLines);
        int smooth_offset = (offset * 0.15) + (last_offset * 0.85);
        last_offset = smooth_offset;

        // -------------------------- Driving Logic --------------------------
        if (validLines.size() >= 2) {
            if (abs(smooth_offset) <= dead_zone) car_forward();
            else if (smooth_offset < -dead_zone) car_turn_left();
            else car_turn_right();
        }
        else if (validLines.size() == 1) {
            float k = getLineParams(validLines[0]).first;
            last_valid_slope = k;
            if (k < -2.2)      car_turn_left();
            else if (k > 2.2)  car_turn_right();
            else               car_forward();
        }
        else {
            if (last_valid_slope < -2)      car_turn_right();
            else if (last_valid_slope > 2)  car_turn_left();
            else                              car_forward();
        }

        cv::imshow("Lane Tracking", frame);
        if (cv::waitKey(1) == 27) break;
    }

    car_stop();
    cap.release();
    cv::destroyAllWindows();
    pigpio_stop(pi);
    return 0;
}