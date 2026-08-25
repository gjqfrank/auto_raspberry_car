#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

// 矩形信息：顶点 + 颜色（0=红 1=绿）
struct RectInfo {
    vector<Point> contour;
    int color; // 0=RED  1=GREEN
};

// 识别所有红色、绿色矩形，并返回颜色
vector<RectInfo> findAllColorRect(Mat& src)
{
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    // ========== 红色掩膜 ==========
    Mat red_mask1, red_mask2, red_mask;
    inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), red_mask1);
    inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), red_mask2);
    red_mask = red_mask1 | red_mask2;

    // ========== 绿色掩膜 ==========
    Mat green_mask;
    inRange(hsv, Scalar(35, 50, 50), Scalar(90, 255, 255), green_mask);

    // ========== 分别找轮廓，判断颜色 ==========
    vector<RectInfo> result;

    // 找红色矩形
    vector<vector<Point>> red_contours;
    findContours(red_mask, red_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (auto& cnt : red_contours) {
        double area = contourArea(cnt);
        if (area < 350) continue;
        double peri = arcLength(cnt, true);
        vector<Point> approx;
        approxPolyDP(cnt, approx, 0.05 * peri, true);
        if (approx.size() == 4) {
            result.push_back({approx, 0});
        }
    }

    // 找绿色矩形
    vector<vector<Point>> green_contours;
    findContours(green_mask, green_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (auto& cnt : green_contours) {
        double area = contourArea(cnt);
        if (area < 350) continue;
        double peri = arcLength(cnt, true);
        vector<Point> approx;
        approxPolyDP(cnt, approx, 0.05 * peri, true);
        if (approx.size() == 4) {
            result.push_back({approx, 1});
        }
    }

    return result;
}

int main()
{
    namedWindow("红绿矩形识别+颜色标注", WINDOW_NORMAL);

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "摄像头打开失败！" << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    Mat frame;
    vector<RectInfo> allRects;
    auto last_check = system_clock::now();
    const float interval = 0.5f;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        auto now = system_clock::now();
        auto delta = duration_cast<milliseconds>(now - last_check).count();

        if (delta >= interval * 1000) {
            allRects = findAllColorRect(frame);
            last_check = now;

            cout << "\n==== 识别到 " << allRects.size() << " 个矩形 ====" << endl;
            for (int i=0; i<allRects.size(); i++) {
                string color_str = (allRects[i].color == 0) ? "红色矩形" : "绿色矩形";
                cout << "[" << i+1 << "] " << color_str << endl;
            }
        }

        // ========== 绘制矩形 + 颜色文字 ==========
        for (auto& rect : allRects) {
            // 画矩形框
            polylines(frame, rect.contour, true, Scalar(0,255,0), 2);

            // 计算中心点
            Moments m = moments(rect.contour);
            int cx = m.m10 / m.m00;
            int cy = m.m01 / m.m00;

            // 显示颜色文字
            string text = (rect.color == 0) ? "RED" : "GREEN";
            Scalar text_color = (rect.color == 0) ? Scalar(0,0,255) : Scalar(0,255,0);
            putText(frame, text, Point(cx-20, cy), 
                FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2);
        }

        imshow("红绿矩形识别+颜色标注", frame);
        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}