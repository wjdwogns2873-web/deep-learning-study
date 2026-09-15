#include "engine.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

int main()
{
    std::string modelPath = "../best.onnx";
    const char* logid = "yolo_inference";
    const char* provider = "CPU";

    YoloInferencer inferencer(modelPath, logid, provider);

    std::string imagePath = "../fruit_0002.png";
    cv::Mat image = cv::imread(imagePath);

    if (image.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    std::vector<Detection> detections = inferencer.infer(image, 0.1, 0.5);

    for (const auto& detection : detections) {
        cv::rectangle(image, detection.box, cv::Scalar(0, 255, 0), 2);
        std::cout << "Detection: Class=" << detection.class_id << ", Confidence=" << detection.confidence
            << ", x=" << detection.box.x << ", y=" << detection.box.y
            << ", width=" << detection.box.width << ", height=" << detection.box.height << std::endl;
    }

    cv::imshow("output", image);
    cv::waitKey(0);

    return 0;
}