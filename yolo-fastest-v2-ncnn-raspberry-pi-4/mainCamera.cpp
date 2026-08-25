// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "yolo-fastestv2.h"

yoloFastestv2 yoloF2;

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
    for (size_t i = 0; i < boxes.size(); i++) {
        char text[256];
        sprintf(text, "%s %.1f%%", class_names[boxes[i].cate + 1], boxes[i].score * 100);

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

int main(int argc, char** argv)
{
    const std::string defaultUrl = "http://127.0.0.1:8080/?action=stream";
    const std::string streamUrl = (argc > 1) ? argv[1] : defaultUrl;

    yoloF2.init(false); // no GPU
    yoloF2.loadModel("yolo-fastestv2-opt.param", "yolo-fastestv2-opt.bin");

    cv::VideoCapture cap(streamUrl);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Unable to open stream URL: " << streamUrl << std::endl;
        std::cerr << "Usage: ./mainCamera [mjpg_stream_url]" << std::endl;
        return 1;
    }

    std::cout << "Streaming from: " << streamUrl << std::endl;
    std::cout << "Press ESC to quit." << std::endl;

    float fpsWindow[16];
    int fpsIdx = 0;
    for (int i = 0; i < 16; i++) fpsWindow[i] = 0.0f;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "ERROR: Empty frame from stream." << std::endl;
            break;
        }

        const auto tBegin = std::chrono::steady_clock::now();

        std::vector<TargetBox> boxes;
        yoloF2.detection(frame, boxes);
        draw_objects(frame, boxes);

        const auto tEnd = std::chrono::steady_clock::now();
        const float ms = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tBegin).count();
        if (ms > 0.0f) fpsWindow[(fpsIdx++) & 0x0F] = 1000.0f / ms;

        float fps = 0.0f;
        for (int i = 0; i < 16; i++) fps += fpsWindow[i];
        fps /= 16.0f;

        cv::putText(frame, cv::format("FPS %0.2f", fps), cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255));

        cv::imshow("mainCamera", frame);
        const char key = static_cast<char>(cv::waitKey(5));
        if (key == 27) break;
    }

    return 0;
}
