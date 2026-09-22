#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct Detection {
    cv::Rect box;
    float confidence;
    int classId;
};

class YoloORT {
public:
    // fp16Mode가 true이면 FP16 모델 입력 텐서로 동작
    YoloORT(const std::string& modelPath, bool isFP16 = false, const cv::Size& inputSize = cv::Size(640, 640));
    ~YoloORT() = default;

    std::vector<Detection> detect(const cv::Mat& frame, float confThreshold = 0.25f, float nmsThreshold = 0.45f);

private:
    void preprocess(const cv::Mat& frame, std::vector<float>& inputTensorValues);
    void preprocessFP16(const cv::Mat& frame, std::vector<Ort::Float16_t>& inputValues);
    std::vector<Detection> postprocess(std::vector<Ort::Value>& outputTensors, float confThreshold, float nmsThreshold);
    
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;

    bool isFP16Mode;
    cv::Size inputImageSize;

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<const char*> inputNamePtrs;
    std::vector<const char*> outputNamePtrs;

    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShape;
};