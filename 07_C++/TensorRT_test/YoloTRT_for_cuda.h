#ifndef YOLO_TRT_H
#define YOLO_TRT_H

#include "preprocess.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <filesystem>

// 탐지된 박스 정보를 표현할 Detection 구조체를 선언합니다.
struct Detection {
    cv::Rect box; // xmin, ymin, width, height를 담을 box
    float confidence; // 확신도를 담을 confidence
    int classId; // 클래스 아이디를 담을 classId
};

// TensorRT에서는 Logger 구현이 필수적입니다.
// TensorRT 엔진 빌드 및 추론 과정에서 발생하는 경고나 에러 메시지를 출력하기 위한 Logger입니다.
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class YoloTRT {
public:
    YoloTRT(const std::string& enginePath);
    ~YoloTRT();

    // 카메라 스레드에서 각자 병렬로 호출할 수 있도록 static으로 선언합니다.
    static void preprocessSingle(const cv::Mat& srcImg, std::vector<float>& outBuffer);

    // 이미 전처리가 완료된 float 벡터 묶음을 받아 복사 후 곧바로 GPU 추론만 수행합니다.
    std::vector<std::vector<Detection>> detectBatchPreprocessed(
        const std::vector<std::vector<float>>& batchBuffers, 
        const std::vector<cv::Size>& origSizes, 
        float confThreshold = 0.25f, 
        float nmsThreshold = 0.45f
    );

    // 실제로 추론할 detect 함수를 선언합니다.
    // 이미지, conf 임계값과 nms 임계값을 매개변수로 받아서 std::vector<Detection>을 반환합니다.
    std::vector<Detection> detect(const cv::Mat& srcImg, float confThresh = 0.25f, float nmsThresh = 0.45f);

    // 배치 단위로 추론 결과가 나오기 때문에, std::vector<Detection>을 한 번더 벡터로 감싸서
    // std::vector<std::vector<Detection>>이 됩니다.
    // cv::Mat도 벡터로 들어갑니다.
    std::vector<std::vector<Detection>> detectBatch(
        const std::vector<cv::Mat>& srcImgs, 
        float confThreshold = 0.25f, 
        float nmsThreshold = 0.45f
    );

    void inferOnly();

    std::vector<std::vector<Detection>> detectBatchGPU(
        const std::vector<cv::Mat>& srcImgs, 
        float confThreshold = 0.25f, 
        float nmsThreshold = 0.45f
    );

    // 1장당 요소 개수 상수를 외부(카메라 스레드)에서도 버퍼 크기 예약에 쓸 수 있도록 public으로 이동합니다.
    static constexpr size_t kSingleInputElements = 3 * 640 * 640;
private:
    static constexpr int kMaxBatchSize = 4;

    TRTLogger gLogger;
    // nvinfer1이 인터페이스라서 인스턴스를 직접 생성할 수 없기 때문에 오직 포인터로만 다뤄야 합니다.
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    
    // 전처리 함수입니다. 640 640 리사이즈, HWC to CHW, BGR to RGB 등의 전처리 작업을 수행합니다.
    // 배치 인덱스를 받아 h_input의 알맞은 위치에 기록하도록 수정합니다.
    void preprocess(const cv::Mat& srcImg, int batchIdx = 0);

    // 후처리 함수입니다. 원본이미지로 복원하는 과정, nms 작업 등이 있습니다.
    // 여기서도 배치 인덱스를 받아서 h_output의 알맞은 위치에 기록하도록 수정합니다.
    std::vector<Detection> postprocess(int origW, int origH, float confThresh, float nmsThresh, int batchIdx = 0);

    cudaStream_t stream = nullptr;

    // Device(GPU)의 입출력 포인터(주소값)입니다.
    void* d_input = nullptr;
    void* d_output = nullptr;

    // Host(CPU)의 입출력 포인터(주소값)입니다.
    float* h_input = nullptr;
    float* h_output = nullptr;

    // 1장당 요소 개수 상수 (포인터 오프셋 계산용)
    static constexpr size_t kSingleOutputElements = 84 * 8400;

    // 최대 배치 크기 기준으로 메모리 할당 크기 확장
    size_t maxInputSize = kMaxBatchSize * kSingleInputElements * sizeof(float);
    size_t maxOutputSize = kMaxBatchSize * kSingleOutputElements * sizeof(float);

    // 입출력 텐서의 메모리 크기입니다.
    // size_t inputSize = 1 * 3 * 640 * 640 * sizeof(float);
    // size_t outputSize = 1 * 84 * 8400 * sizeof(float);

    // 동적으로 메모리 할당 및 해제를 방지하기 위해 최초 한 번만 생성합니다.
    std::vector<cv::Rect> boxes; // 탐지된 네모박스 정보(cv::Rect)를 담기 위해 std::vector<cv::Rect>를 선언합니다.
    std::vector<float> confidences; // 네모박스들의 확신도를 담기 위해 std::vector<float>을 선언합니다.
    std::vector<int> classIds; // 네모박스들의 클래스 아이디를 담기 위해 std::vector<int>를 선언합니다.
    std::vector<int> indices; // conf, nms 임계값을 넘은 네모박스들의 임계값을 담을 std::vector<int>를 선언합니다.


    static constexpr size_t kMaxSrcBytesPerImage = 1920 * 1080 * 3;
    uint8_t* d_src = nullptr;
    uint8_t* h_src = nullptr;
};

#endif // YOLO_TRT_H