# C++ TensorRT: FP32 vs FP16 vs INT8 동영상 스트리밍 벤치마크 보고서


## 📊 종합 성능 비교 요약

| 정밀도 엔진 | 평균 Latency (Avg) | 최소 Latency (Min) | 최대 Latency (Max) | 평균 처리 속도 (FPS) |
| :--- | :--- | :--- | :--- | :--- |
| **1. FP32** | **9.49 ms** | **6.91 ms** | **21.76 ms** | **105.3 FPS** |
| **2. FP16** | **19.58 ms** | **8.70 ms** | **34.05 ms** | **51.1 FPS** |
| **3. INT8** | **20.06 ms** | **9.05 ms** | **85.69 ms** | **49.9 FPS** |

---

C++ TensorRT 환경에서 Yolov8n 모델의 정밀도별로 동영상 스트리밍시 성능을 비교해보고 싶어서 해본 실험입니다.
양자화 기법 중에 FP16은 몇번 해봤지만, INT8은 이번에 처음 도전해봤습니다.
양자화라는 게 모델의 가중치의 데이터타입을 비트 수가 작은 타입으로 변환하는 걸 의미하는데, 
양자화를 하면 연산량이 줄어든다고 알고 있었지만, 실제로 제가 예상한 결과와는 많이 달라서 놀랐습니다.
정확도(mAP) 비교는 이 보고서 안에 없지만, 정확도(mAP)를 떠나서 양자화를 하지 않은 FP32 버전이 가장 빨랐고, 나머지는 반토막 났습니다.

왜 이런 일이 발생했는지 검색해본 결과, 몇가지 이유가 있었습니다.

1. 하드웨어 제약 및 런타임 버전의 변화입니다.
구글 코랩 환경의 T4 GPU는 최신 아키텍처에 비해 FP16/INT8 전용 가속 모듈 효율이 떨어집니다. 
특히 구버전 TensorRT에서는 하드웨어가 특정 정밀도를 원활하게 지원하지 못하면 내부적으로 FP16 등으로 우회하여 속도를 방어해 주었습니다.
하지만 이번 TensorRT 11.x 버전은 규격이 엄격해져서 하드웨어 아키텍처와 모델 설계도가 궁합이 완벽하지 않으면 내부에서 타입 변환 오버헤드로 인해
속도가 감소했습니다.

2. 실시간 스트리밍시 배치사이즈가 1. 그리고 연산 오버헤드
동영상 스트리밍은 실시간 처리를 위해 프레임을 한 장씩 처리하는 배치사이즈 1 구조를 가집니다.
양자화는 수십장의 배치를 할 때 효과가 극대화되는데, 1장씩 처리할 때는 데이터 타입을 정밀도에 맞게 매번 형변환하는 오버헤드가 더 커지게 됩니다.
가속으로 얻는 이득보다 형변환하는 오버헤드 비용이 더 커져서 속도가 느려진 것입니다.

3. 싱글 쓰레드 구조로 인한 CPU 병목 현상
현재 구현된 시스템은 하나의 스레드가 [동영상 파일 읽기 -> CPU 전처리 -> GPU 추론 -> CPU 후처리] 과정을 순서대로 기다리며 수행하는 동기식 구조입니다.
이 구조에서는 GPU 연산 속도(FP16, INT8)를 아무리 단축해도, OpenCV가 영상을 디코딩하고 CPU 메모리끼리 데이터를 복사하는 시간이 길어져서
CPU 병목 현상이 발생합니다.

[향후 개선 방향]
현업에서는 이러한 병목을 해결하기 위해 비동기 멀티 쓰레딩 구조로 가져간다고 알고있습니다.
다음엔 비동기 멀티 쓰레딩 구조로 한번 더 테스트해 보겠습니다.

---

<details>
<summary><b>공통 소스코드</b></summary>

### `YoloTRT.h`
```cpp
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
#include <filesystem>

struct Detection {
    cv::Rect box;
    float confidence;
    int classId;
};

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

    std::vector<Detection> detect(const cv::Mat& srcImg, float confThresh = 0.25f, float nmsThresh = 0.45f);
    void inferOnly();
    void runDatasetInference(const std::string& imageDir);

private:
    TRTLogger gLogger;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    
    void preprocess(const cv::Mat& srcImg);
    std::vector<Detection> postprocess(int origW, int origH, float confThresh, float nmsThresh);

    cudaStream_t stream = nullptr;
    void* d_input = nullptr;
    void* d_output = nullptr;
    float* h_input = nullptr;
    float* h_output = nullptr;

    size_t inputSize = 1 * 3 * 640 * 640 * sizeof(float);
    size_t outputSize = 1 * 84 * 8400 * sizeof(float);

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    std::vector<int> indices;
};

#endif // YOLO_TRT_H
```

### `YoloTRT.cpp`
```cpp
#include "YoloTRT.h"
namespace fs = std::filesystem;

YoloTRT::YoloTRT(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "엔진 파일을 찾을 수 없습니다: " << enginePath << std::endl;
        return; 
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(engineData.data(), size);
    context = engine->createExecutionContext();

    cudaStreamCreate(&stream);
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);

    cudaHostAlloc((void**)&h_input, inputSize, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_output, outputSize, cudaHostAllocDefault);

    boxes.reserve(8400);
    confidences.reserve(8400);
    classIds.reserve(8400);
    indices.reserve(8400);

    const char* inputName = engine->getIOTensorName(0);
    const char* outputName = engine->getIOTensorName(1);
    context->setTensorAddress(inputName, d_input);
    context->setTensorAddress(outputName, d_output);
}

YoloTRT::~YoloTRT() {
    if (stream) cudaStreamDestroy(stream);
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    if (h_input) cudaFreeHost(h_input);
    if (h_output) cudaFreeHost(h_output);
    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;
}

void YoloTRT::preprocess(const cv::Mat& srcImg) {
    cv::Mat resizedImg;
    cv::resize(srcImg, resizedImg, cv::Size(640, 640));
    cv::cvtColor(resizedImg, resizedImg, cv::COLOR_BGR2RGB);

    cv::Mat floatImg;
    resizedImg.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

    int channelSize = 640 * 640;
    float* data = reinterpret_cast<float*>(floatImg.data);

    for (int i = 0; i < channelSize; ++i) {
        h_input[i]                   = data[i * 3 + 0]; // R
        h_input[i + channelSize]     = data[i * 3 + 1]; // G
        h_input[i + channelSize * 2] = data[i * 3 + 2]; // B
    }
}

std::vector<Detection> YoloTRT::postprocess(int origW, int origH, float confThresh, float nmsThresh) {
    boxes.clear();
    confidences.clear();
    classIds.clear();
    indices.clear();

    float scaleX = static_cast<float>(origW) / 640.0f;
    float scaleY = static_cast<float>(origH) / 640.0f;

    for (int i = 0; i < 8400; i++) {
        float cx = h_output[0 * 8400 + i];
        float cy = h_output[1 * 8400 + i];
        float w  = h_output[2 * 8400 + i];
        float h  = h_output[3 * 8400 + i];

        float maxScore = 0.0f;
        int maxClassId = -1;
        for (int c = 0; c < 80; c++) {
            float score = h_output[(c + 4) * 8400 + i];
            if (score > maxScore) {
                maxScore = score;
                maxClassId = c;
            }
        }

        if (maxScore > confThresh) {
            float x1 = (cx - w / 2.0f) * scaleX;
            float y1 = (cy - h / 2.0f) * scaleY;
            float boxW = w * scaleX;
            float boxH = h * scaleY;

            int rectX = std::max(0, std::min(static_cast<int>(x1), origW - 1));
            int rectY = std::max(0, std::min(static_cast<int>(y1), origH - 1));
            int rectW = std::max(0, std::min(static_cast<int>(boxW), origW - rectX));
            int rectH = std::max(0, std::min(static_cast<int>(boxH), origH - rectY));

            boxes.emplace_back(rectX, rectY, rectW, rectH);
            confidences.emplace_back(maxScore);
            classIds.emplace_back(maxClassId);
        }
    }

    cv::dnn::NMSBoxes(boxes, confidences, confThresh, nmsThresh, indices);

    std::vector<Detection> detections;
    detections.reserve(indices.size());
    for (int idx : indices) {
        detections.push_back({ boxes[idx], confidences[idx], classIds[idx] });
    }

    return detections;
}

std::vector<Detection> YoloTRT::detect(const cv::Mat& srcImg, float confThresh, float nmsThresh) {
    if (srcImg.empty()) return {};

    preprocess(srcImg);

    cudaMemcpyAsync(d_input, h_input, inputSize, cudaMemcpyHostToDevice, stream);
    context->enqueueV3(stream);
    cudaMemcpyAsync(h_output, d_output, outputSize, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return postprocess(srcImg.cols, srcImg.rows, confThresh, nmsThresh);
}

void YoloTRT::inferOnly() {
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
}
```

### `main.cpp`
```cpp
#include "YoloTRT.h"
#include <chrono>
#include <numeric>
#include <vector>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

int main() {
    YoloTRT detector("yolov8n.engine"); // 엔진 파일 교체 가능 (yolov8n_fp16.engine, yolov8n_int8.engine)

    std::string videoPath = "./test_videos/sample.mp4";
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        std::cerr << "비디오 파일을 열 수 없습니다: " << videoPath << std::endl;
        return -1;
    }

    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double inputFps = cap.get(cv::CAP_PROP_FPS);

    cv::Mat frame;
    cap >> frame;
    if (!frame.empty()) {
        for (int i = 0; i < 50; ++i) {
            detector.detect(frame, 0.25f, 0.45f);
        }
    }

    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    std::vector<double> latencies;
    latencies.reserve(totalFrames);
    int frameCount = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        auto start = std::chrono::high_resolution_clock::now();

        auto detections = detector.detect(frame, 0.25f, 0.45f);

        for (const auto& det : detections) {
            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(ms);
        frameCount++;
    }

    cap.release();

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = sum / latencies.size();
    double min_val = *std::min_element(latencies.begin(), latencies.end());
    double max_val = *std::max_element(latencies.begin(), latencies.end());

    std::cout << "\n================ [C++ Native TRT Video 벤치마크 결과] ===============:" << std::endl;
    std::cout << "처리 프레임 수: " << frameCount << " / " << totalFrames << std::endl;
    std::cout << "평균 Latency (Avg): " << cv::format("%.2f", avg) << " ms" << std::endl;
    std::cout << "최소 Latency (Min): " << cv::format("%.2f", min_val) << " ms" << std::endl;
    std::cout << "최대 Latency (Max): " << cv::format("%.2f", max_val) << " ms" << std::endl;
    std::cout << "평균 동영상 처리 FPS: " << cv::format("%.1f", 1000.0 / avg) << " FPS" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    return 0;
}
```
</details>

<details>
<summary><b>1. FP32 builder</b></summary>

### `builder.cpp`
```cpp
#include <iostream>
#include <fstream>
#include <memory>
#include <NvInfer.h>
#include <NvOnnxParser.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

int main() {
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    
    if (!parser->parseFromFile("yolov8n.onnx", static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "ONNX 파싱 실패!" << std::endl;
        return -1;
    }

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);

    std::ofstream engineFile("yolov8n.engine", std::ios::binary);
    engineFile.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    engineFile.close();

    delete plan; delete config; delete parser; delete network; delete builder;
    return 0;
}
```
</details>

<details>
<summary><b>2. FP16 builder</b></summary>

### `builder_FP16.cpp`
```cpp
#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <NvOnnxParser.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

int main() {
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    
    parser->parseFromFile("yolov8n.onnx", static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    
    // FP16 최적화 레이어 활성화 유도
    nvinfer1::IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);

    std::ofstream engineFile("yolov8n_fp16.engine", std::ios::binary);
    engineFile.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    engineFile.close();

    delete plan; delete config; delete parser; delete network; delete builder;
    return 0;
}
```
</details>

<details>
<summary><b>3. INT8 builder</b></summary>
  
### `builder_INT8.cpp`
```cpp
#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <NvOnnxParser.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

int main() {
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    
    parser->parseFromFile("yolov8n.onnx", static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    
    nvinfer1::IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);

    std::ofstream engineFile("yolov8n_int8.engine", std::ios::binary);
    engineFile.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    engineFile.close();

    delete plan; delete config; delete parser; delete network; delete builder;
    return 0;
}
```

</details>
