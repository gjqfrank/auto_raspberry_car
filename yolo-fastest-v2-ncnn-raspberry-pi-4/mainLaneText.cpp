/* Use openCV to do Lane Detection

*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
const unsigned char hough_threshold = 150;
const unsigned char hough_min_line_length = 80;
const unsigned char hough_max_line_gap = 40;


static void zeroLowComponentPixels(cv::Mat& frame)
{
    for (int row = 0; row < frame.rows; ++row) {
        cv::Vec3b* pixel = frame.ptr<cv::Vec3b>(row);
        for (int col = 0; col < frame.cols; ++col) {
            if (pixel[col][0] < blue_threshold || pixel[col][1] <       green_threshold || pixel[col][2] < red_threshold) {
                pixel[col] = cv::Vec3b(0, 0, 0);
            }
        }
    }
}

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
    cv::Mat gray;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Can not read frame from camera" << std::endl;
            break;
        }

        cv::Mat frameCopy = frame.clone();

        // Canny edge detection,
        // Convert the BGR frame to grayscale. Colour information is not needed for edge detection and processing a single channel is 3× faster.
        // cv::cvtColor(frameCopy, frameCopy, cv::COLOR_BGR2GRAY);

        // Apply Gaussian blur to suppress noise. A 5×5 kernel is a good starting point for typical road footage.
        // cv::GaussianBlur(frameCopy, frameCopy, cv::Size(5, 5), 0);

        // cv::Canny(frameCopy, frameCopy, canny_low_threshold, canny_high_threshold);   


        cv::putText(frameCopy, cv::format("size %d", 18), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);


        cv::imshow("Camera Lane Detection", frameCopy);


        if(cv::waitKey(5) == 27) { // ESC key to exit
            break;
        }

    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

