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

// 검출 결과를 담는 구조체
struct Detection {
    cv::Rect box;
    float confidence;
    int classId;
};

// TensorRT ILogger 로거
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class YoloTRT {
public:
    // 생성자 및 소멸자 (RAII 자원 관리)
    YoloTRT(const std::string& enginePath);
    ~YoloTRT();

    // End-to-End 검출 함수 (단순 Resize + 최적화 전/후처리)
    std::vector<Detection> detect(const cv::Mat& srcImg, float confThresh = 0.25f, float nmsThresh = 0.45f);

    // 순수 추론 전용 메서드
    void inferOnly();

private:
    // 최적화된 내부 전처리/후처리 함수
    void preprocess(const cv::Mat& srcImg);
    std::vector<Detection> postprocess(int origW, int origH, float confThresh, float nmsThresh);

private:
    TRTLogger gLogger;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;

    cudaStream_t stream = nullptr;

    // GPU Device 버퍼
    void* d_input = nullptr;
    void* d_output = nullptr;

    // CPU Pinned Host 버퍼
    float* h_input = nullptr;
    float* h_output = nullptr;

    // 텐서 메모리 크기
    size_t inputSize = 1 * 3 * 640 * 640 * sizeof(float);
    size_t outputSize = 1 * 84 * 8400 * sizeof(float);

    // 후처리 버퍼 재사용 (동적 메모리 할당 해제 방지)
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    std::vector<int> indices;
};

#endif // YOLO_TRT_H