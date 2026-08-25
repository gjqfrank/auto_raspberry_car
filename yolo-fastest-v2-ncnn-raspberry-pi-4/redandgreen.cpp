#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <thread>

// ==================== PWM 驱动库 ====================
#include <pigpiod_if2.h>

using namespace cv;
using namespace std;
using namespace chrono;

// ==================== PWM 电机引脚定义 ====================
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
const int PWM_FREQ = 2000;
const int DUTY_STRAIGHT = 400000;

//const int PWM_FREQ = 0;
// const int DUTY_STRAIGHT = 0;


// ==================== PWM 全局句柄 ====================
int pi;

// ==================== 矩形识别结构体 =====================
struct RectInfo
{
    vector<Point> contour;
    bool is_green;
};

// ==================== 状态机枚举 =====================
enum CarState {
    STATE_STOP,   // 停止状态
    STATE_GO      // 运行状态
};

// ==================== PWM 电机初始化 ====================
void motor_init() {
    pi = pigpio_start(NULL, NULL);
    if (pi < 0) {
        cerr << "pigpio 连接失败！请确认pigpiod服务已启动：sudo pigpiod" << endl;
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

// ==================== PWM 电机控制 ====================
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

// ===================== 高精度颜色识别 =====================
vector<RectInfo> detect_color_rect(Mat& frame)
{
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    Mat red1, red2, red_mask;
    inRange(hsv, Scalar(0, 150, 100), Scalar(10, 255, 255), red1);
    inRange(hsv, Scalar(170, 150, 100), Scalar(180, 255, 255), red2);
    red_mask = red1 | red2;

    Mat green_mask;
    inRange(hsv, Scalar(50, 100, 100), Scalar(75, 255, 255), green_mask);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5));
    morphologyEx(red_mask, red_mask, MORPH_CLOSE, kernel);
    morphologyEx(green_mask, green_mask, MORPH_CLOSE, kernel);

    vector<RectInfo> result;
    vector<vector<Point>> contours;

    findContours(red_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (auto& cnt : contours)
    {
        double area = contourArea(cnt);
        if (area < 500) continue;
        vector<Point> approx;
        approxPolyDP(cnt, approx, 0.04 * arcLength(cnt, true), true);
        if (approx.size() == 4) result.push_back({approx, false});
    }

    contours.clear();
    findContours(green_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (auto& cnt : contours)
    {
        double area = contourArea(cnt);
        if (area < 500) continue;
        vector<Point> approx;
        approxPolyDP(cnt, approx, 0.04 * arcLength(cnt, true), true);
        if (approx.size() == 4) result.push_back({approx, true});
    }
    return result;
}

// ===================== 主程序（状态机 + 新逻辑） =====================
int main()
{
    motor_init();
    car_stop();

    VideoCapture cap(0);
    cap.set(CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CAP_PROP_FRAME_HEIGHT, 240);
    if (!cap.isOpened()) {
        cerr << "摄像头打开失败！" << endl;
        car_stop();
        pigpio_stop(pi);
        return -1;
    }

    namedWindow("状态机视觉小车", WINDOW_NORMAL);
    Mat frame;

    // ===================== 状态机核心变量 =====================
    CarState current_state = STATE_STOP;  // 初始：停止
    int green_count = 0;                  // 绿色计数
    int red_count = 0;                    // 红色计数
    auto start_time = system_clock::now();// 计时起点
    const int CHECK_TIME = 100;           // 检查窗口：100ms = 0.1秒
    const int TRIGGER_COUNT = 3;          // 需要连续3次

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        // 每帧都检测颜色
        vector<RectInfo> rects = detect_color_rect(frame);
        bool has_green = false, has_red = false;
        for (auto& r : rects) {
            if (r.is_green) has_green = true;
            else has_red = true;
        }

        // 计算时间差
        auto now = system_clock::now();
        int delta_ms = duration_cast<milliseconds>(now - start_time).count();

        // ===================== 状态机逻辑 =====================
        if (current_state == STATE_STOP)
        {
            // STOP 状态：统计绿色
            if (has_green) green_count++;

            // 0.1秒时间窗到了
            if (delta_ms >= CHECK_TIME)
            {
                if (green_count >= TRIGGER_COUNT)
                {
                    current_state = STATE_GO;
                    car_forward();
                    cout << "✅ 0.1秒内3次绿色 → 进入GO状态，小车前进" << endl;
                }
                // 重置
                green_count = 0;
                start_time = now;
            }
        }
        else if (current_state == STATE_GO)
        {
            // GO 状态：统计红色
            if (has_red) red_count++;

            // 0.1秒时间窗到了
            if (delta_ms >= CHECK_TIME)
            {
                if (red_count >= TRIGGER_COUNT)
                {
                    current_state = STATE_STOP;
                    car_stop();
                    cout << "🛑 0.1秒内3次红色 → 进入STOP状态，小车停止" << endl;
                }
                // 重置
                red_count = 0;
                start_time = now;
            }
        }

        // 绘制画面
        for (auto& r : rects)
        {
            polylines(frame, r.contour, true, Scalar(0,255,0), 2);
            string label = r.is_green ? "GREEN" : "RED";
            Scalar color = r.is_green ? Scalar(0,255,0) : Scalar(0,0,255);
            putText(frame, label, Point(r.contour[0].x, r.contour[0].y-10),
                    FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }

        // 显示当前状态
        string state_str = (current_state == STATE_STOP) ? "STATE: STOP" : "STATE: GO";
        putText(frame, state_str, Point(20, 40), FONT_HERSHEY_SIMPLEX, 1,
                Scalar(255,255,255), 2);

        imshow("状态机视觉小车", frame);
        if (waitKey(1) == 27) break;
    }

    car_stop();
    cap.release();
    destroyAllWindows();
    pigpio_stop(pi);
    return 0;
}