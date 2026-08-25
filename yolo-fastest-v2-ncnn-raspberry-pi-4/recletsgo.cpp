#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <gpiod.h>
#include <unistd.h>
#include <pigpiod_if2.h>

using namespace cv;
using namespace std;
using namespace chrono;

// ===================== 引脚定义（完全沿用你原来的BCM编号）=====================
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


const int PWM_FREQ = 100;
const int DUTY_STRAIGHT = 250000;   // Ö±ÐÐ»ù×¼Õ¼¿Õ±È
const int DUTY_LEFT_STR = 400000;  // ×óÂÖÇ¿
const int DUTY_RIGHT_STR = 250000; // ÓÒÂÖÈõ
// ===================== 电机底层控制 =====================
void motor_init() {
    pi = pigpio_start(NULL, NULL);
    if (pi < 0) {
        std::cerr << "pigpio Á¬½ÓÊ§°Ü£¡" << std::endl;
        exit(1);
    }

    // ·½ÏòÒý½Å
    set_mode(pi, LF_FWD, PI_OUTPUT);
    set_mode(pi, LF_BWD, PI_OUTPUT);
    set_mode(pi, LB_FWD, PI_OUTPUT);
    set_mode(pi, LB_BWD, PI_OUTPUT);
    set_mode(pi, RF_FWD, PI_OUTPUT);
    set_mode(pi, RF_BWD, PI_OUTPUT);
    set_mode(pi, RB_FWD, PI_OUTPUT);
    set_mode(pi, RB_BWD, PI_OUTPUT);

    // PWMÒý½Å
    set_mode(pi, LF_PWM, PI_OUTPUT);
    set_mode(pi, LB_PWM, PI_OUTPUT);
    set_mode(pi, RF_PWM, PI_OUTPUT);
    set_mode(pi, RB_PWM, PI_OUTPUT);
}

// ==================== PWM µç»ú¶¯×÷º¯Êý ====================
void car_stop() {
    // ·½ÏòÇåÁã
    gpio_write(pi, LF_FWD, 0); gpio_write(pi, LF_BWD, 0);
    gpio_write(pi, LB_FWD, 0); gpio_write(pi, LB_BWD, 0);
    gpio_write(pi, RF_FWD, 0); gpio_write(pi, RF_BWD, 0);
    gpio_write(pi, RB_FWD, 0); gpio_write(pi, RB_BWD, 0);

    // PWMÍ£Ö¹
    hardware_PWM(pi, LF_PWM, 0, 0);
    hardware_PWM(pi, LB_PWM, 0, 0);
    hardware_PWM(pi, RF_PWM, 0, 0);
    hardware_PWM(pi, RB_PWM, 0, 0);
}

void car_forward() {
    // ·½Ïò£ºÈ«²¿Ç°½ø
    gpio_write(pi, LF_FWD, 1); gpio_write(pi, LF_BWD, 0);
    gpio_write(pi, LB_FWD, 1); gpio_write(pi, LB_BWD, 0);
    gpio_write(pi, RF_FWD, 1); gpio_write(pi, RF_BWD, 0);
    gpio_write(pi, RB_FWD, 1); gpio_write(pi, RB_BWD, 0);

    // ×óÓÒÍ¬ËÙ
    hardware_PWM(pi, LF_PWM, PWM_FREQ, DUTY_STRAIGHT);
    hardware_PWM(pi, LB_PWM, PWM_FREQ, DUTY_STRAIGHT);
    hardware_PWM(pi, RF_PWM, PWM_FREQ, DUTY_STRAIGHT);
    hardware_PWM(pi, RB_PWM, PWM_FREQ, DUTY_STRAIGHT);
}

void car_turn_left() {
    // ·½Ïò£ºÈ«²¿Ç°½ø
    gpio_write(pi, LF_FWD, 1); gpio_write(pi, LF_BWD, 0);
    gpio_write(pi, LB_FWD, 1); gpio_write(pi, LB_BWD, 0);
    gpio_write(pi, RF_FWD, 1); gpio_write(pi, RF_BWD, 0);
    gpio_write(pi, RB_FWD, 1); gpio_write(pi, RB_BWD, 0);

    // ×óÂÖ¿ì¡¢ÓÒÂÖÂý£¨²îËÙ×ó×ª£©
    hardware_PWM(pi, LF_PWM, PWM_FREQ, DUTY_LEFT_STR);
    hardware_PWM(pi, LB_PWM, PWM_FREQ, DUTY_LEFT_STR);
    hardware_PWM(pi, RF_PWM, PWM_FREQ, DUTY_RIGHT_STR);
    hardware_PWM(pi, RB_PWM, PWM_FREQ, DUTY_RIGHT_STR);
}

void car_turn_right() {
    // ·½Ïò£ºÈ«²¿Ç°½ø
    gpio_write(pi, LF_FWD, 1); gpio_write(pi, LF_BWD, 0);
    gpio_write(pi, LB_FWD, 1); gpio_write(pi, LB_BWD, 0);
    gpio_write(pi, RF_FWD, 1); gpio_write(pi, RF_BWD, 0);
    gpio_write(pi, RB_FWD, 1); gpio_write(pi, RB_BWD, 0);

    // ÓÒÂÖ¿ì¡¢×óÂÖÂý£¨²îËÙÓÒ×ª£©
    hardware_PWM(pi, LF_PWM, PWM_FREQ, DUTY_RIGHT_STR);
    hardware_PWM(pi, LB_PWM, PWM_FREQ, DUTY_RIGHT_STR);
    hardware_PWM(pi, RF_PWM, PWM_FREQ, DUTY_LEFT_STR);
    hardware_PWM(pi, RB_PWM, PWM_FREQ, DUTY_LEFT_STR);
}

// ===================== 矩形识别结构体 =====================
struct RectInfo
{
    vector<Point> contour;
    bool is_green; // true=绿色，false=红色
};

// ===================== 红绿颜色矩形检测 =====================
vector<RectInfo> detect_color_rect(Mat& frame)
{
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    // 红色HSV范围（适配环境光线）
    Mat red1, red2, red_mask;
    inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), red1);
    inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), red2);
    red_mask = red1 | red2;

    // 优化后的绿色HSV范围，大幅提升识别灵敏度
    Mat green_mask;
    inRange(hsv, Scalar(35, 50, 50), Scalar(90, 255, 255), green_mask);

    vector<RectInfo> result;
    vector<vector<Point>> contours;

    // 提取所有红色矩形
    findContours(red_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (auto& cnt : contours)
    {
        double area = contourArea(cnt);
        if (area < 400) continue;

        vector<Point> approx;
        approxPolyDP(cnt, approx, 0.05 * arcLength(cnt, true), true);
        if (approx.size() == 4)
        {
            result.push_back({approx, false});
        }
    }

    // 提取所有绿色矩形
    contours.clear();
    findContours(green_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (auto& cnt : contours)
    {
        double area = contourArea(cnt);
        if (area < 400) continue;

        vector<Point> approx;
        approxPolyDP(cnt, approx, 0.05 * arcLength(cnt, true), true);
        if (approx.size() == 4)
        {
            result.push_back({approx, true});
        }
    }

    return result;
}

// ===================== 主程序 =====================
int main()
{
    // 初始化硬件+先保持停止
    motor_init();
    car_stop();

    // 打开摄像头
    VideoCapture cap(0);
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    namedWindow("红绿视觉自动小车", WINDOW_NORMAL);

    Mat frame;
    auto last_update = system_clock::now();
    const int detect_interval = 500; // 0.5秒刷新一次检测

    bool has_red = false;
    bool has_green = false;

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        auto now = system_clock::now();
        auto delta = duration_cast<milliseconds>(now - last_update).count();

        // 定时检测+小车决策
        if (delta >= detect_interval)
        {
            vector<RectInfo> rects = detect_color_rect(frame);
            last_update = now;

            has_red = false;
            has_green = false;

            for (auto& r : rects)
            {
                if (r.is_green) has_green = true;
                else has_red = true;
            }

            // 核心控制逻辑：红色优先
            if (has_red)
            {
                car_stop();
                cout << "🔴 识别红色矩形 → 小车停止" << endl;
            }
            else if (has_green)
            {
                car_forward();
                cout << "🟢 识别绿色矩形 → 小车前进" << endl;
            }
            else
            {
                car_stop();
                cout << "⚪ 无目标 → 小车停止" << endl;
            }
        }

        // 画面绘制+文字标注（文字固定在矩形框上方）
        vector<RectInfo> rects_draw = detect_color_rect(frame);
        for (auto& r : rects_draw)
        {
            polylines(frame, r.contour, true, Scalar(0, 255, 0), 2);

            // 找到矩形最顶部Y坐标，文字放在上方
            int top_y = 480;
            for (auto& p : r.contour)
            {
                if (p.y < top_y) top_y = p.y;
            }

            string label = r.is_green ? "GREEN" : "RED";
            Scalar text_color = r.is_green ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
            putText(frame, label, Point(r.contour[0].x, top_y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
        }

        imshow("红绿视觉自动小车", frame);

        // 按下ESC退出程序
        if (waitKey(1) == 27) break;
    }

    // 退出前安全停车+释放资源
    car_stop();
    cap.release();
    destroyAllWindows();
    gpiod_chip_close(chip);

    return 0;
}