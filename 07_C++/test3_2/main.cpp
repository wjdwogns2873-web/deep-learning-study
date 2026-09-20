#include "engine.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

int main()
{
    std::string modelPath = "../best.onnx";
    std::string logid = "yolo_inference";
    std::string provider = "CPU";

    YOLOv8Detector detector(modelPath, logid, provider);

    std::string imagePath = "../fruit_0002.png";
    cv::Mat image = cv::imread(imagePath);

    if (image.empty()) {
        std::cerr << "Error: Unable to load iamge!" << std::endl;
        return -1;
    }

    std::vector<Detection> detections = detector.infer(image, 0.4, 0.5);

    cv::Mat resultImage = detector.drawDetections(image, detections);

    cv::imshow("output", resultImage);
    cv::waitKey(0);

    return 0;
}