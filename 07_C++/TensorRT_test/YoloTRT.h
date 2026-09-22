#ifndef YOLO_TRT_H
#define YOLO_TRT_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

struct Detection {
    cv::Rect box;
    float conf;
    int classId;
};

struct LetterboxInfo {
    float scale;
    int dw;
    int dh;
};

// TensorRT Logger
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class YoloTRT {
public:
    // 생성자 및 소멸자 (RAII 기반 자동 메모리 관리)
    YoloTRT(const std::string& enginePath);
    ~YoloTRT();

    std::vector<Detection> detect(const cv::Mat& srcImg, float confThresh = 0.25f, float nmsThresh = 0.45f);

private:
    TRTLogger gLogger;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;

    cudaStream_t stream = nullptr;

    void* d_input = nullptr;
    void* d_output = nullptr;

    size_t inputSize = 1 * 3 * 640 * 640 * sizeof(float);
    size_t outputSize = 1 * 84 * 8400 * sizeof(float);

    std::vector<float> inputHost;
    std::vector<float> outputHost;

    void preprocessLetterbox(const cv::Mat& srcImg, LetterboxInfo& info);
    std::vector<Detection> postprocess(const LetterboxInfo& info, int origW, int origH, float confThresh, float nmsThresh);
};

#endif // YOLO_TRT_H