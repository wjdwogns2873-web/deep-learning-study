#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// 검출 결과를 깔끔하게 묶어 전달할 구조체
struct Detection {
    cv::Rect box;
    float confidence;
    int classId;
};

class YOLOv8Detector {
public:
    // 생성자 및 소멸자
    YOLOv8Detector(const std::string& modelPath, float confThresh = 0.25f, float nmsThresh = 0.45f);
    ~YOLOv8Detector() = default;

    // 메인 추론 메서드
    std::vector<Detection> detect(const cv::Mat& frame);

    // 시각화 메서드
    void drawResults(cv::Mat& frame, const std::vector<Detection>& detections, float fps = 0.0f);

private:
    // 내부 전처리 / 후처리 메서드 (캡슐화)
    Ort::Value preprocess(const cv::Mat& frame, std::vector<float>& inputTensorValues);
    std::vector<Detection> postprocess(const cv::Mat& frame, float* floatArray);

    // ONNX Runtime 자원 관리
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo;

    // 모델 매개변수
    int inputWidth = 640;
    int inputHeight = 640;
    int numAnchors = 8400;
    int numClasses = 3;

    float confThreshold;
    float nmsThreshold;

    const char* inputNames[1] = {"images"};
    const char* outputNames[1] = {"output0"};
    std::vector<int64_t> inputShape = {1, 3, 640, 640};
};