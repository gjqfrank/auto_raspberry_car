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
#include <stdio.h>
#include <iostream>
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
    yoloF2.init(false); // no GPU
    yoloF2.loadModel("yolo-fastestv2-opt.param", "yolo-fastestv2-opt.bin");

    cv::Mat frame = cv::imread("./parking.jpg");
    if (frame.empty()) {
        frame = cv::imread("./parking.jpg");
    }

    if (frame.empty()) {
        std::cerr << "ERROR: Unable to open ./parking_tiny.jpg or ./parking.jpg" << std::endl;
        return 1;
    }

    std::vector<TargetBox> boxes;
    yoloF2.detection(frame, boxes);
    draw_objects(frame, boxes);

    std::cout << "Detections: " << boxes.size() << std::endl;
    cv::imwrite("./parking_result.jpg", frame);

    cv::imshow("mainImg", frame);
    cv::waitKey(0);
    return 0;
}
