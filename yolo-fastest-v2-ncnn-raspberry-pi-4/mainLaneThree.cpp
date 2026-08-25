/* Use openCV to do Lane Detection

*/


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "yolo-fastestv2.h"

yoloFastestv2 yoloF2;

const unsigned char red_threshold = 200;
const unsigned char green_threshold = 200;
const unsigned char blue_threshold = 200;

const unsigned int IMG_WIDTH = 352;
const unsigned int IMG_HEIGHT = 288;
const cv::Point mask_left_bottom(0, IMG_HEIGHT);
const cv::Point mask_right_bottom(IMG_WIDTH-1, IMG_HEIGHT);
// const cv::Point mask_left_top(IMG_WIDTH / 3, IMG_HEIGHT/2);
// const cv::Point mask_right_top(IMG_WIDTH *2/ 3, IMG_HEIGHT/2);
const cv::Point mask_apex(IMG_WIDTH / 2, IMG_HEIGHT / 2 - 10);

const unsigned char canny_low_threshold = 180;
const unsigned char canny_high_threshold = 240;

const unsigned char hough_rho = 1;
const double hough_theta = CV_PI / 180;
const unsigned char hough_threshold = 35;
const unsigned char hough_min_line_length = 25;
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

static void draw_objects(cv::Mat& cvImg, const std::vector<TargetBox>& boxes)
{
    unsigned char id;

    for (size_t i = 0; i < boxes.size(); i++) {
        char text[256];
        sprintf(text, "%s %.1f%%", class_names[boxes[i].cate + 1], boxes[i].score * 100);

        id = boxes[i].cate + 1;
        // 如果不是人，车，自行车，摩托车，红绿灯，停车标志等相关的物体，就不画框了
        if (id != PERSON_CLASS_ID && id != BICYCLE_CLASS_ID && id != CAR_CLASS_ID && id != MOTORBIKE_CLASS_ID && id != TRAFFIC_LIGHT_CLASS_ID && id != STOP_SIGN_CLASS_ID) {
            continue; // skip irrelevant classes
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

// static void zeroLowComponentPixels(cv::Mat& frame)
// {
//     for (int row = 0; row < frame.rows; ++row) {
//         cv::Vec3b* pixel = frame.ptr<cv::Vec3b>(row);
//         for (int col = 0; col < frame.cols; ++col) {
//             if (pixel[col][0] < blue_threshold || pixel[col][1] <       green_threshold || pixel[col][2] < red_threshold) {
//                 pixel[col] = cv::Vec3b(0, 0, 0);
//             }
//         }
//     }
// }

void region_of_interest(cv::Mat& img) {
    // Define a polygon that covers the lower half of the image (where the road is likely to be)
    std::vector<cv::Point> vertices;
    vertices.push_back(mask_left_bottom);
    vertices.push_back(mask_right_bottom);
    vertices.push_back(mask_apex);
    cv::Mat mask = cv::Mat::zeros(img.size(), img.type());
    cv::fillConvexPoly(mask, vertices, cv::Scalar(255, 255, 255));
    cv::bitwise_and(img, mask, img);    
}

int main(int argc, char** argv)
{
    // const std::string defaultUrl = "http://127.0.0.1:8080/?action=stream";
    // const std::string streamUrl = (argc > 1) ? argv[1] : defaultUrl;

    // 初始化yoloF2模型，加载参数和权重文件
    yoloF2.init(false); // no GPU
    yoloF2.loadModel("yolo-fastestv2-opt.param", "yolo-fastestv2-opt.bin");

    // /dev/video0
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 352);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 288);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

        // 检查是否成功打开
    if (!cap.isOpened()) {
        std::cerr << "Can not open camera" << std::endl;
        return -1;
    }

    cv::Mat frame;
    // 测试帧率的
    float fpsWindow[16];
    int fpsIdx = 0;
    for (int i = 0; i < 16; i++) fpsWindow[i] = 0.0f;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Can not read frame from camera" << std::endl;
            break;
        }
        // framecopy用来进行道路识别
        cv::Mat frameCopy = frame.clone();


        // Canny edge detection,
        // Convert the BGR frame to grayscale. Colour information is not needed for edge detection and processing a single channel is 3× faster.
        cv::cvtColor(frameCopy, frameCopy, cv::COLOR_BGR2GRAY);

        // Apply Gaussian blur to suppress noise. A 5×5 kernel is a good starting point for typical road footage.
        cv::GaussianBlur(frameCopy, frameCopy, cv::Size(5, 5), 0);

        cv::Canny(frameCopy, frameCopy, canny_low_threshold, canny_high_threshold);   

        // mask region, ROI
        // vertices of a triangle
        region_of_interest(frameCopy);

        // Set pixels to black when any BGR component is below 200.
        // zeroLowComponentPixels(frameCopy);

        // hough transform to detect lane lines
        std::vector<cv::Vec4i> lines;
        // Parameters: image, output vector, rho, theta, threshold
        cv::HoughLinesP(frameCopy, lines, hough_rho, hough_theta, hough_threshold, hough_min_line_length, hough_max_line_gap);

        cv::putText(frameCopy, cv::format("line num: %d", lines.size()), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

        // 在frame上画线
        for (size_t i = 0; i < lines.size(); i++) {
            cv::Vec4i l = lines[i];
            cv::line(frame, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
        }

        // 在frame上画识别物体的框框
        // 暂时只识别人体，红绿灯，car等，其它无关的物体不识别
        const auto tBegin = std::chrono::steady_clock::now();

        std::vector<TargetBox> boxes;
        yoloF2.detection(frame, boxes); 
        // 在frame上画识别物体的框框和标签
        draw_objects(frame, boxes);

        const auto tEnd = std::chrono::steady_clock::now();
        const float ms = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tBegin).count();
        if (ms > 0.0f) fpsWindow[(fpsIdx++) & 0x0F] = 1000.0f / ms;

        float fps = 0.0f;
        for (int i = 0; i < 16; i++) fps += fpsWindow[i];
        fps /= 16.0f;

        cv::putText(frame, cv::format("FPS %0.2f", fps), cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255));

        cv::imshow("Camera Lane Detection", frameCopy);
        cv::imshow("Original", frame);

        
        if(cv::waitKey(5) == 27) { // ESC key to exit
            break;
        }

    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

