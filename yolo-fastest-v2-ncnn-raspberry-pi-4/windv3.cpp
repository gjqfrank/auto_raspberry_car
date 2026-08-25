#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <signal.h>
#include <chrono>
#include <deque>
#include "yolo-fastestv2.h"
#include <pigpiod_if2.h>

// ==================== 电机引脚定义（你的硬件）====================
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

// ==================== 核心参数（弯道专用）====================
const int PWM_FREQ = 100;
const int BASE_SPEED = 200000;     // 正常速度
const int SLOW_SPEED = 140000;     // 弯道减速速度
const int WHEEL_BASE = 15;
const int LOOK_AHEAD = 80;

// ==================== 弯道判断参数 ====================
const int DEVIATION_THRESHOLD = 20;  // 偏离超过20
const int CHECK_COUNT = 5;           // 连续5次
const double CHECK_TIME = 0.5;       // 0.5秒内

// ==================== 退出控制 ====================
bool need_exit = false;
int pi;
yoloFastestv2 yoloF2;
const int W = 320, H = 240;

// ==================== 全局变量（新增平滑 + 弯道判断）====================
std::deque<int> err_history;            // 偏移历史（用于滤波）
std::deque<double> last_check_times;    // 时间记录 ✅ 已修复
int current_speed = BASE_SPEED;
bool is_corner = false;

void car_stop() {
    gpio_write(pi, LF_FWD, 0); gpio_write(pi, LF_BWD, 0);
    gpio_write(pi, LB_FWD, 0); gpio_write(pi, LB_BWD, 0);
    gpio_write(pi, RF_FWD, 0); gpio_write(pi, RF_BWD, 0);
    gpio_write(pi, RB_FWD, 0); gpio_write(pi, RB_BWD, 0);
    hardware_PWM(pi, LF_PWM, 0, 0); hardware_PWM(pi, LB_PWM, 0, 0);
    hardware_PWM(pi, RF_PWM, 0, 0); hardware_PWM(pi, RB_PWM, 0, 0);
}

void handle_stop(int sig) {
    need_exit = true;
    car_stop();
}

void motor_init() {
    pi = pigpio_start(NULL, NULL);
    set_mode(pi, LF_FWD, PI_OUTPUT); set_mode(pi, LF_BWD, PI_OUTPUT);
    set_mode(pi, LB_FWD, PI_OUTPUT); set_mode(pi, LB_BWD, PI_OUTPUT);
    set_mode(pi, RF_FWD, PI_OUTPUT); set_mode(pi, RF_BWD, PI_OUTPUT);
    set_mode(pi, RB_FWD, PI_OUTPUT); set_mode(pi, RB_BWD, PI_OUTPUT);
    set_mode(pi, LF_PWM, PI_OUTPUT); set_mode(pi, LB_PWM, PI_OUTPUT);
    set_mode(pi, RF_PWM, PI_OUTPUT); set_mode(pi, RB_PWM, PI_OUTPUT);
}

void set_diff_drive(int left, int right) {
    left  = std::max(120000, std::min(left, 250000));
    right = std::max(120000, std::min(right, 250000));

    gpio_write(pi, LF_FWD, 1); gpio_write(pi, LF_BWD, 0);
    gpio_write(pi, LB_FWD, 1); gpio_write(pi, LB_BWD, 0);
    gpio_write(pi, RF_FWD, 1); gpio_write(pi, RF_BWD, 0);
    gpio_write(pi, RB_FWD, 1); gpio_write(pi, RB_BWD, 0);

    hardware_PWM(pi, LF_PWM, PWM_FREQ, left);
    hardware_PWM(pi, LB_PWM, PWM_FREQ, left);
    hardware_PWM(pi, RF_PWM, PWM_FREQ, right);
    hardware_PWM(pi, RB_PWM, PWM_FREQ, right);
}

// 滑动平均滤波（中心点平滑）
int smooth_error(int raw_err) {
    err_history.push_back(raw_err);
    if (err_history.size() > 8) {
        err_history.pop_front();
    }
    int sum = 0;
    for (int e : err_history) sum += e;
    return sum / err_history.size();
}

// 弯道检测 + 减速
void check_corner(int err) {
    auto now = std::chrono::steady_clock::now();
    double now_sec = std::chrono::duration<double>(now.time_since_epoch()).count();

    last_check_times.push_back(now_sec);
    if (last_check_times.size() > CHECK_COUNT) {
        last_check_times.pop_front();
    }

    static int dev_cnt = 0;
    if (abs(err) > DEVIATION_THRESHOLD) dev_cnt++;
    else dev_cnt = 0;

    bool time_ok = false;
    if (last_check_times.size() == CHECK_COUNT) {
        double dt = last_check_times.back() - last_check_times.front();
        time_ok = (dt <= CHECK_TIME);
    }

    if (dev_cnt >= CHECK_COUNT && time_ok) {
        is_corner = true;
        current_speed = SLOW_SPEED;
    } else if (dev_cnt == 0) {
        is_corner = false;
        current_speed = BASE_SPEED;
    }
}

void pure_pursuit(int center_err) {
    int max_err = 100;
    center_err = std::max(-max_err, std::min(center_err, max_err));

    double k = 0.7;
    int left  = current_speed - center_err * k;
    int right = current_speed + center_err * k;

    set_diff_drive(left, right);
}

std::vector<cv::Point> get_lane_points(cv::Mat &binary) {
    std::vector<cv::Point> pts;
    for(int y=H-80; y<H; y++){
        for(int x=0; x<W; x++){
            if(binary.at<uchar>(y,x) > 128)
                pts.emplace_back(x,y);
        }
    }
    return pts;
}

cv::Mat fit_poly(std::vector<cv::Point> &pts, int order=2) {
    if(pts.size() < 5) return cv::Mat::zeros(3,1,CV_64F);
    cv::Mat A(pts.size(), 3, CV_64F);
    cv::Mat b(pts.size(), 1, CV_64F);
    for(int i=0; i<pts.size(); i++){
        double x = pts[i].x, y = pts[i].y;
        A.at<double>(i,0) = y*y;
        A.at<double>(i,1) = y;
        A.at<double>(i,2) = 1;
        b.at<double>(i,0) = x;
    }
    cv::Mat coeff;
    cv::solve(A, b, coeff, cv::DECOMP_NORMAL);
    return coeff;
}

int calc_x(cv::Mat &coeff, int y) {
    double a = coeff.at<double>(0), b=coeff.at<double>(1), c=coeff.at<double>(2);
    return (int)(a*y*y + b*y + c);
}

int main() {
    signal(SIGINT, handle_stop);
    motor_init();
    car_stop();

    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, H);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));

    cv::Mat frame;
    while(!need_exit) {
        cap >> frame;
        if(frame.empty()) break;

        cv::Mat gray, blur, canny;
        cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        GaussianBlur(gray, blur, cv::Size(5,5), 0);
        Canny(blur, canny, 50, 100);

        auto pts = get_lane_points(canny);
        auto coeff = fit_poly(pts);

        int target_y = H - 20;
        int cx = calc_x(coeff, target_y);
        int center = W / 2;
        int raw_err = cx - center;

        int smooth_err = smooth_error(raw_err);
        check_corner(raw_err);

        if(pts.size() > 20)
            pure_pursuit(smooth_err);
        else
            car_stop();

        cv::circle(frame, cv::Point(cx, target_y), 6, cv::Scalar(0,255,0), -1);
        cv::line(frame, cv::Point(center, H), cv::Point(center, H-60), cv::Scalar(0,0,255), 2);

        cv::putText(frame, cv::format("Err: %d", smooth_err),
                    cv::Point(10,25), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,0),1);
        cv::putText(frame, is_corner ? "CORNER - SLOW" : "NORMAL - RUN",
                    cv::Point(10,50), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,255,0),1);

        cv::imshow("edge", canny);
        cv::imshow("lane", frame);
        
        if (cv::waitKey(10) == 27) {
            car_stop();
            break;
        }
    }

    car_stop();
    cap.release();
    pigpio_stop(pi);
    cv::destroyAllWindows();
    return 0;
}