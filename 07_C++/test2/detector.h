#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <utility>
#include <vector>
#include <string>

#include "utils.h"

class YOLODetector
{
public:
    explicit YOLODetector(std::nullptr_t) {};
    YOLODetector(const std::string& modelPath, const bool& isGPU, const cv::Size& inputSize);

    std::vector<Detection> detect(cv::Mat &image, const float& confThreshold, const float& iouThreshold);

private:
    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{nullptr};
    Ort::Session session{nullptr};

    void preprocessing(cv::Mat &image, float*& blob, std::vector<int64_t>& inputTensorShape);
    std::vector<Detection> postprocessing(const cv::Size& resizedImageShape, 
                                        const cv::Size& originalImageShape, 
                                        std::vector<Ort::Value>& outputTensors, 
                                        const float& confThreshold, const float& iouThreshold);
    static void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses, 
                                float& bestConf, int& bestClassId);
    
    // ONNX Runtime 최신 API 규격에 맞추어 std::string으로 변경
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    
    bool isDynamicInputShape{};
    cv::Size2f inputImageShape;
};