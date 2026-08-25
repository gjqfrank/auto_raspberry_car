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

// write a function do Region of Interest (ROI) to mask out the sky and surroundings to reduce false positives dramatically.
// 去掉上边一半的面积，裁掉
void region_of_interest(cv::Mat& img, cv::Mat& mask) {
    // Define a polygon that covers the lower half of the image (where the road is likely to be)
    std::vector<cv::Point> vertices;
    int img_width = img.cols;
    int img_height = img.rows;
    vertices.push_back(cv::Point(0, img_height)); // Bottom-left corner
    vertices.push_back(cv::Point(img_width, img_height)); // Bottom-right corner        
    vertices.push_back(cv::Point(img_width, img_height * 0.5)); // Top-right corner
    vertices.push_back(cv::Point(0, img_height * 0.5)); // Top-left corner      
    cv::fillConvexPoly(mask, vertices, cv::Scalar(255));
    cv::bitwise_and(img, mask, img);
}

// write a function to do Hough Transform for Lane Lines
void hough_transform(cv::Mat& img) {
    std::vector<cv::Vec4i> lines;
    // Parameters: image, output vector, rho, theta, threshold
    cv::HoughLinesP(img, lines, 1, CV_PI / 180, 150, 80, 40);
    for (size_t i = 0; i < lines.size(); i++) {
        cv::Vec4i l = lines[i];
        cv::line(img, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255), 3, cv::LINE_AA);
    }
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

        // Convert the BGR frame to grayscale. Colour information is not needed for edge detection and processing a single channel is 3× faster.
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

        // Apply Gaussian blur to suppress noise. A 5×5 kernel is a good starting point for typical road footage.
        cv::GaussianBlur(frame, frame, cv::Size(5, 5), 1.5,0);

        // Canny edge detection finds sharp intensity gradients, which correspond to lane markings and road boundaries.
        //  Lower threshold = 50, upper = 150 (adjust for your lighting)
        cv::Canny(frame, frame, 60, 180);

        // ROI, Region of Interest
        // Only the lower portion of the frame contains the road ahead. Mask out the sky and surroundings to reduce false positives dramatically.
        cv::Mat mask = cv::Mat::zeros(frame.size(), frame.type());
        region_of_interest(frame, mask);

        // Hough Transform for Lane Lines
        // The Hough Transform detects lines in the edge-detected image. It works by transforming points in Cartesian space to a parameter space and finding accumulations that correspond to lines.
        hough_transform(frame);

        cv::imshow("Camera Lane Detection", frame);


        if(cv::waitKey(20) == 27) { // ESC key to exit
            break;
        }

    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

