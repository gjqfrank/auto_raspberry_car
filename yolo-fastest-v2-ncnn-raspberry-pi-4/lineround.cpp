#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "yolo-fastestv2.h"
#include <gpiod.h>
#include <unistd.h>

// ==================== 电机引脚（你原来的）=====================
#define RF_FWD  22
#define RF_BWD  27
#define LF_FWD  5
#define LF_BWD  6
#define RB_FWD  13
#define RB_BWD  19
#define LB_FWD  12
#define LB_BWD  16

struct gpiod_line *rf_fwd, *rf_bwd;
struct gpiod_line *lf_fwd, *lf_bwd;
struct gpiod_line *rb_fwd, *rb_bwd;
struct gpiod_line *lb_fwd, *lb_bwd;
struct gpiod_chip *chip;

// ==================== 全局防抖变量 ====================
int last_offset = 0;

yoloFastestv2 yoloF2;

const int IMG_W = 320;
const int IMG_H = 240;

// ==================== 函数声明 ====================
void set_pin(struct gpiod_line* l, int v);
void soft_pwm(struct gpiod_line *line, int duty);
void motor_init();
void car_stop();
void go_straight();
void turn_soft_left();
void turn_soft_right();

// ==================== PWM 慢速控制 ====================
void soft_pwm(struct gpiod_line *line, int duty) {
    if (duty <= 0) {
        gpiod_line_set_value(line, 0);
        return;
    }
    gpiod_line_set_value(line, 1);
    usleep(duty * 100);
    gpiod_line_set_value(line, 0);
    usleep((100 - duty) * 100);
}

void set_pin(struct gpiod_line* l, int v) {
    gpiod_line_set_value(l, v);
}

void motor_init() {
    chip = gpiod_chip_open_by_name("gpiochip0");
    if (!chip) {
        std::cerr << "GPIO init fail\n";
        exit(1);
    }

    rf_fwd = gpiod_chip_get_line(chip, RF_FWD);
    rf_bwd = gpiod_chip_get_line(chip, RF_BWD);
    lf_fwd = gpiod_chip_get_line(chip, LF_FWD);
    lf_bwd = gpiod_chip_get_line(chip, LF_BWD);
    rb_fwd = gpiod_chip_get_line(chip, RB_FWD);
    rb_bwd = gpiod_chip_get_line(chip, RB_BWD);
    lb_fwd = gpiod_chip_get_line(chip, LB_FWD);
    lb_bwd = gpiod_chip_get_line(chip, LB_BWD);

    gpiod_line_request_output(rf_fwd, "m", 0);
    gpiod_line_request_output(rf_bwd, "m", 0);
    gpiod_line_request_output(lf_fwd, "m", 0);
    gpiod_line_request_output(lf_bwd, "m", 0);
    gpiod_line_request_output(rb_fwd, "m", 0);
    gpiod_line_request_output(rb_bwd, "m", 0);
    gpiod_line_request_output(lf_fwd, "m", 0);
    gpiod_line_request_output(lb_bwd, "m", 0);
}

void car_stop() {
    set_pin(rf_fwd,0); set_pin(rf_bwd,0);
    set_pin(lf_fwd,0); set_pin(lf_bwd,0);
    set_pin(rb_fwd,0); set_pin(rb_bwd,0);
    set_pin(lb_fwd,0); set_pin(lb_bwd,0);
}

// 直道慢速
void go_straight() {
    set_pin(rf_bwd,0); set_pin(lf_bwd,0); set_pin(rb_bwd,0); set_pin(lb_bwd,0);
    soft_pwm(rf_fwd,17); soft_pwm(lf_fwd,17);
    soft_pwm(rb_fwd,17); soft_pwm(lb_fwd,17);
}

// 小左拐（弯道用）
void turn_soft_left() {
    set_pin(rf_bwd,0); set_pin(lf_bwd,0); set_pin(rb_bwd,0); set_pin(lb_bwd,0);
    soft_pwm(rf_fwd,17); soft_pwm(rb_fwd,17);
    soft_pwm(lf_fwd,10); soft_pwm(lb_fwd,10);
}

// 小右拐（弯道用）
void turn_soft_right() {
    set_pin(rf_bwd,0); set_pin(lf_bwd,0); set_pin(rb_bwd,0); set_pin(lb_bwd,0);
    soft_pwm(lf_fwd,17); soft_pwm(lb_fwd,17);
    soft_pwm(rf_fwd,10); soft_pwm(rb_fwd,10);
}

// ==================== 黑白化提取白线 ====================
cv::Mat binarize_white(cv::Mat& frame) {
    cv::Mat gray, thresh;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, thresh, 200, 255, cv::THRESH_BINARY);
    return thresh;
}

// ==================== ROI 掩码 ====================
void roi_mask(cv::Mat& img) {
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
    std::vector<cv::Point> p = {{0,IMG_H}, {IMG_W,IMG_H}, {IMG_W,120}, {0,120}};
    cv::fillConvexPoly(mask, p, 255);
    cv::bitwise_and(img, mask, img);
}

// ==================== 滑动窗口 ====================
std::vector<cv::Point> sliding_window(cv::Mat& binary, bool is_left) {
    int win_h = IMG_H / 10;
    int win_w = 20;
    int margin = 15;
    std::vector<cv::Point> pts;

    int last_x = is_left ? IMG_W/4 : 3*IMG_W/4;
    for (int y = IMG_H - win_h; y > 80; y -= win_h) {
        cv::Rect r(last_x - margin, y, win_w + 2*margin, win_h);
        if (r.x < 0) r.x = 0;
        if (r.x + r.width > IMG_W) r.width = IMG_W - r.x;

        std::vector<cv::Point> non_zero;
        cv::findNonZero(binary(r), non_zero);
        if (!non_zero.empty()) {
            cv::Scalar m = cv::mean(non_zero);
            last_x = r.x + (int)m[0];
        }
        pts.emplace_back(last_x, y + win_h/2);
    }
    return pts;
}

// ==================== 简易曲线拟合（兼容所有OpenCV）====================
cv::Point2f fit_curve(std::vector<cv::Point>& pts, int y) {
    if (pts.size() < 3) return {0.0f, (float)y};

    float sumY = 0, sumYY = 0, sumX = 0, sumXY = 0, sumXYY =0;
    int n = pts.size();
    for (auto& p : pts) {
        float yi = p.y;
        float xi = p.x;
        sumY += yi;
        sumYY += yi*yi;
        sumX += xi;
        sumXY += xi*yi;
    }

    float a = (n*sumXY - sumX*sumY) / (n*sumYY - sumY*sumY + 1e-6);
    float b = (sumX - a*sumY)/n;
    float x = a*y + b;
    return {x, (float)y};
}

// ==================== 绘制弯道车道 + 中心线 ====================
int draw_curve_lane(cv::Mat& frame) {
    cv::Mat bin = binarize_white(frame);
    roi_mask(bin);

    auto left = sliding_window(bin, true);
    auto right = sliding_window(bin, false);
    if (left.empty() || right.empty()) return 9999;

    cv::Point2f p_l = fit_curve(left, IMG_H-10);
    cv::Point2f p_r = fit_curve(right, IMG_H-10);

    cv::line(frame, p_l, fit_curve(left, 100), cv::Scalar(255,255,255), 2);
    cv::line(frame, p_r, fit_curve(right, 100), cv::Scalar(255,255,255), 2);

    int cx = (int)((p_l.x + p_r.x) / 2);
    cv::Point center_bot(cx, IMG_H);
    cv::Point center_top( (int)((fit_curve(left,100).x + fit_curve(right,100).x)/2), 100 );
    cv::line(frame, center_bot, center_top, cv::Scalar(0,255,0), 3);

    return cx - IMG_W/2;
}

// ==================== 主函数 ====================
int main() {
    motor_init();
    car_stop();

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "cam fail\n";
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, IMG_W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, IMG_H);

    while (1) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) continue;

        // 弯道车道识别
        int offset = draw_curve_lane(frame);

        // 防抖
        float smooth = 0.85f;
        int ofs = (int)(offset*(1-smooth) + last_offset*smooth);
        last_offset = ofs;

        // 循迹过弯
        if (offset == 9999)
            car_stop();
        else if (abs(ofs) < 30)
            go_straight();
        else if (ofs < 0)
            turn_soft_left();
        else
            turn_soft_right();

        cv::imshow("lane", frame);
        if (cv::waitKey(1) == 27) {
            car_stop();
            break;
        }
    }

    cap.release();
    gpiod_chip_close(chip);
    return 0;
}